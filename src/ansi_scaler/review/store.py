from __future__ import annotations

import json
import os
import sqlite3
import threading
import warnings
from pathlib import Path
from typing import Any, Iterable

from ansi_scaler.config import RunConfig
from ansi_scaler.review.models import ReviewEvent


SCHEMA_VERSION = 2


class ReviewStore:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.review_dir = config.run_dir / "reviews"
        self.database_path = self.review_dir / "index.sqlite3"
        self.annotation_path = self.review_dir / "annotations.jsonl"
        self.revision = 0
        self.review_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._initialise()
        self.refresh_manifests(force=True)
        self.replay_annotations()

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def _initialise(self) -> None:
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            self.connection.executescript(
                "DROP TABLE IF EXISTS records; DROP TABLE IF EXISTS errors; "
                "DROP TABLE IF EXISTS review_events; DROP TABLE IF EXISTS ingest_sources;"
            )
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                parent_id TEXT,
                artifact TEXT,
                concept_id TEXT,
                concept_name TEXT,
                kit_id TEXT,
                role TEXT,
                source_file TEXT NOT NULL,
                source_line INTEGER NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS records_parent_stage ON records(parent_id, stage);
            CREATE INDEX IF NOT EXISTS records_stage ON records(stage);
            CREATE INDEX IF NOT EXISTS records_filters ON records(kit_id, role, concept_id);

            CREATE TABLE IF NOT EXISTS errors (
                source_file TEXT NOT NULL,
                source_line INTEGER NOT NULL,
                parent_id TEXT,
                payload TEXT NOT NULL,
                PRIMARY KEY(source_file, source_line)
            );

            CREATE TABLE IF NOT EXISTS ingest_sources (
                path TEXT PRIMARY KEY,
                mtime_ns INTEGER NOT NULL,
                size INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS review_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                run TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                snapshot_id TEXT NOT NULL,
                outcome TEXT,
                issue_code TEXT,
                introduced_by TEXT,
                target_stage TEXT,
                target_output_id TEXT,
                supersedes TEXT,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS reviews_sample ON review_events(sample_id, recorded_at);
            CREATE INDEX IF NOT EXISTS reviews_snapshot ON review_events(snapshot_id, recorded_at);
            """
        )
        self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.connection.commit()

    def refresh_manifests(self, *, force: bool = False) -> int:
        changed = 0
        paths = sorted(self.config.manifest_dir.glob("*.jsonl"))
        with self._lock, self.connection:
            for path in paths:
                stat = path.stat()
                state = self.connection.execute(
                    "SELECT mtime_ns, size FROM ingest_sources WHERE path = ?", (str(path),)
                ).fetchone()
                if not force and state and state["mtime_ns"] == stat.st_mtime_ns and state["size"] == stat.st_size:
                    continue
                self._replace_manifest(path)
                self.connection.execute(
                    "INSERT OR REPLACE INTO ingest_sources(path, mtime_ns, size) VALUES (?, ?, ?)",
                    (str(path), stat.st_mtime_ns, stat.st_size),
                )
                changed += 1
        if changed:
            self.revision += 1
        return changed

    def _replace_manifest(self, path: Path) -> None:
        is_error = path.name.endswith(".errors.jsonl")
        self.connection.execute("DELETE FROM errors WHERE source_file = ?", (str(path),))
        if not is_error:
            self.connection.execute("DELETE FROM records WHERE source_file = ?", (str(path),))
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                encoded = json.dumps(record, sort_keys=True, ensure_ascii=False)
                if is_error:
                    self.connection.execute(
                        "INSERT INTO errors(source_file, source_line, parent_id, payload) VALUES (?, ?, ?, ?)",
                        (str(path), line_number, record.get("parent_id"), encoded),
                    )
                    continue
                if not record.get("id") or not record.get("stage"):
                    continue
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO records(
                        id, stage, parent_id, artifact, concept_id, concept_name, kit_id, role,
                        source_file, source_line, payload
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["id"],
                        record["stage"],
                        record.get("parent_id"),
                        record.get("artifact"),
                        record.get("concept_id"),
                        record.get("concept_name"),
                        record.get("kit_id"),
                        record.get("role"),
                        str(path),
                        line_number,
                        encoded,
                    ),
                )

    def replay_annotations(self) -> int:
        if not self.annotation_path.exists():
            return 0
        raw = self.annotation_path.read_bytes()
        lines = raw.splitlines(keepends=True)
        inserted = 0
        with self._lock, self.connection:
            for index, encoded in enumerate(lines):
                if not encoded.strip():
                    continue
                if not encoded.endswith((b"\n", b"\r")) and index == len(lines) - 1:
                    warnings.warn("Ignoring crash-truncated final annotation line", stacklevel=2)
                    continue
                try:
                    event = ReviewEvent.model_validate_json(encoded)
                except Exception as error:
                    raise ValueError(f"Invalid review annotation at line {index + 1}: {error}") from error
                inserted += self._insert_event(event)
        if inserted:
            self.revision += 1
        return inserted

    def append_event(self, event: ReviewEvent) -> None:
        encoded = json.dumps(event.model_dump(mode="json"), sort_keys=True, ensure_ascii=False) + "\n"
        with self._lock:
            descriptor = os.open(self.annotation_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                os.write(descriptor, encoded.encode())
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            with self.connection:
                self._insert_event(event)
            self.revision += 1

    def _insert_event(self, event: ReviewEvent) -> int:
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO review_events(
                event_id, event_type, recorded_at, reviewer, run, sample_id, snapshot_id,
                outcome, issue_code, introduced_by, target_stage, target_output_id, supersedes, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.event_type,
                event.recorded_at.isoformat(),
                event.reviewer,
                event.run,
                event.sample_id,
                event.snapshot_id,
                event.outcome,
                event.issue_code,
                event.introduced_by,
                event.target_stage,
                event.target_output_id,
                event.supersedes,
                json.dumps(event.model_dump(mode="json"), sort_keys=True, ensure_ascii=False),
            ),
        )
        return cursor.rowcount

    def records(self, *, stage: str | None = None, parent_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT payload FROM records WHERE 1 = 1"
        parameters: list[str] = []
        if stage is not None:
            query += " AND stage = ?"
            parameters.append(stage)
        if parent_id is not None:
            query += " AND parent_id = ?"
            parameters.append(parent_id)
        query += " ORDER BY source_line, id"
        with self._lock:
            return [json.loads(row["payload"]) for row in self.connection.execute(query, parameters)]

    def record(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute("SELECT payload FROM records WHERE id = ?", (record_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def review_events(self, *, sample_id: str | None = None) -> list[ReviewEvent]:
        query = "SELECT payload FROM review_events"
        parameters: tuple[str, ...] = ()
        if sample_id is not None:
            query += " WHERE sample_id = ?"
            parameters = (sample_id,)
        query += " ORDER BY recorded_at, event_id"
        with self._lock:
            rows = self.connection.execute(query, parameters).fetchall()
        return [ReviewEvent.model_validate_json(row["payload"]) for row in rows]

    def active_reviews(self) -> list[ReviewEvent]:
        events = self.review_events()
        superseded = {event.supersedes for event in events if event.supersedes}
        active = [event for event in events if event.event_type == "set" and event.event_id not in superseded]
        by_target: dict[tuple[str, str | None], ReviewEvent] = {}
        for event in active:
            key = (self.asset_id(event), event.target_stage if event.schema_version == 3 else None)
            by_target[key] = event
        return list(by_target.values())

    @staticmethod
    def asset_id(event: ReviewEvent) -> str:
        """Return the stable prompt identity, including for legacy raster-keyed events."""
        return event.outputs.get("prompt", event.sample_id)

    def manifest_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self.connection.execute("SELECT stage, COUNT(*) count FROM records GROUP BY stage").fetchall()
        return {row["stage"]: row["count"] for row in rows}

    def errors(self) -> Iterable[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute("SELECT payload FROM errors ORDER BY source_file, source_line").fetchall()
        return (json.loads(row["payload"]) for row in rows)
