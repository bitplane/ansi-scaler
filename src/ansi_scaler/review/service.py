from __future__ import annotations

import getpass
import hashlib
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ansi_scaler.config import RunConfig
from ansi_scaler.identity import stable_id
from ansi_scaler.manifests import resolve_path
from ansi_scaler.review.ansi import PyramidCache, ansi_to_runs
from ansi_scaler.review.models import ReviewEvent, ReviewSubmission
from ansi_scaler.review.store import ReviewStore
from ansi_scaler.stages.generate import SanaGenerator
from ansi_scaler.stages.pyramid import PYRAMID_FORMAT


def _last(records: list[dict[str, Any]], predicate: Any = None) -> dict[str, Any] | None:
    matching = [record for record in records if predicate is None or predicate(record)]
    return matching[-1] if matching else (records[-1] if records else None)


class ReviewService:
    def __init__(self, config: RunConfig, store: ReviewStore | None = None) -> None:
        self.config = config
        self.store = store or ReviewStore(config)
        self.reviewer = os.environ.get("ANSI_SCALER_REVIEWER") or getpass.getuser()
        self._sample_cache: tuple[int, list[dict[str, Any]]] | None = None
        self.pyramid_cache = PyramidCache()

    def close(self) -> None:
        self.pyramid_cache.clear()
        self.store.close()

    def samples(self) -> list[dict[str, Any]]:
        if self._sample_cache is not None and self._sample_cache[0] == self.store.revision:
            return self._sample_cache[1]
        all_records = self.store.records()
        by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
        children: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in all_records:
            by_stage[record["stage"]].append(record)
            if record.get("parent_id"):
                children[(record["parent_id"], record["stage"])].append(record)
        prompts = {record["id"]: record for record in by_stage["prompts"]}
        active_reviews = {self.store.asset_id(event): event for event in self.store.active_reviews()}
        all_reviews: dict[str, list[ReviewEvent]] = defaultdict(list)
        for event in self.store.review_events():
            if event.event_type == "set":
                all_reviews[self.store.asset_id(event)].append(event)

        samples = []
        for prompt in by_stage["prompts"]:
            expected_raster = SanaGenerator(self.config).output_id(prompt)
            raster = _last(children[(prompt["id"], "generate")], lambda item: item["id"] == expected_raster)
            if raster is None:
                continue
            cutout = _last(
                children[(raster["id"], "background")],
                lambda item: item.get("background_settings") == self.config.background.model_dump(mode="json"),
            )
            lod = _last(children[(cutout["id"], "lod")]) if cutout is not None else None
            pyramid = (
                _last(
                    children[(lod["id"], "pyramid")],
                    lambda item: (
                        item.get("chuda_version") == self.config.chuda.version
                        and item.get("pyramid_format") == PYRAMID_FORMAT
                    ),
                )
                if lod is not None
                else None
            )
            classification = (
                _last(
                    children[(cutout["id"], "classify")],
                    lambda item: item.get("vlm_prompt_version") == self.config.vlm.prompt_version,
                )
                if cutout is not None
                else None
            )
            verification = (
                _last(
                    children[(classification["id"], "verify")],
                    lambda item: item.get("llm_prompt_version") == self.config.llm.prompt_version,
                )
                if classification is not None
                else None
            )
            prompt = prompts.get(raster.get("parent_id"), prompt)
            records = {
                key: value
                for key, value in {
                    "prompt": prompt,
                    "generate": raster,
                    "background": cutout,
                    "lod": lod,
                    "pyramid": pyramid,
                    "classify": classification,
                    "verify": verification,
                }.items()
                if value is not None
            }
            outputs = {stage: record["id"] for stage, record in records.items()}
            snapshot_id = stable_id("review-snapshot-v1", raster["id"], outputs)
            asset_id = prompt["id"] if prompt is not None else raster["id"]
            active_review = active_reviews.get(asset_id)
            review = active_review if active_review is not None and active_review.snapshot_id == snapshot_id else None
            machine_decision = verification.get("verification", {}).get("decision") if verification else "missing"
            sample = {
                "sample_id": asset_id,
                "snapshot_id": snapshot_id,
                "outputs": outputs,
                "records": records,
                "prompt": prompt or raster,
                "raster": raster,
                "cutout": cutout,
                "lod": lod,
                "pyramid": pyramid,
                "classification": classification,
                "verification": verification,
                "machine_decision": machine_decision,
                "review": review,
                "history": all_reviews.get(asset_id, []),
                "kit_id": raster.get("location", raster.get("kit_id", "unknown")),
                "role": raster.get("theme", raster.get("role", "unknown")),
                "concept_id": raster.get("specification_id", raster.get("concept_id", "unknown")),
                "concept_name": raster.get("label", raster.get("concept_name", raster.get("concept_id", "Unknown"))),
            }
            sample["conflict"] = self._has_conflict(sample)
            samples.append(sample)
        self._sample_cache = (self.store.revision, samples)
        return samples

    @staticmethod
    def _has_conflict(sample: dict[str, Any]) -> bool:
        current_review = sample["review"]
        if current_review is not None:
            return False
        for old in reversed(sample["history"]):
            if old.snapshot_id == sample["snapshot_id"] or old.outcome not in ("accept", "reject"):
                continue
            decision = sample["machine_decision"]
            decision_conflicts = (old.outcome == "accept" and decision in ("reject", "review")) or (
                old.outcome == "reject" and decision == "accept"
            )
            blamed_stage_changed = bool(
                old.introduced_by and old.outputs.get(old.introduced_by) != sample["outputs"].get(old.introduced_by)
            )
            if decision_conflicts or blamed_stage_changed:
                return True
        return False

    def queue(self, filters: dict[str, str] | None = None, *, include_reviewed: bool = False) -> list[dict[str, Any]]:
        filters = filters or {}
        samples = [sample for sample in self.samples() if self._matches(sample, filters)]
        if not include_reviewed:
            samples = [sample for sample in samples if sample["review"] is None]
        all_samples = self.samples()
        definitive = [sample for sample in all_samples if sample["review"] and sample["review"].outcome != "review"]
        dimensions = ("kit_id", "role", "concept_id", "machine_decision")
        totals = {dimension: Counter(sample[dimension] for sample in all_samples) for dimension in dimensions}
        reviewed = {dimension: Counter(sample[dimension] for sample in definitive) for dimension in dimensions}
        for sample in samples:
            ratios = [
                (reviewed[dimension][sample[dimension]] + 1) / (totals[dimension][sample[dimension]] + 1)
                for dimension in dimensions
            ]
            sample["coverage_score"] = sum(ratios) / len(ratios)
            digest = hashlib.sha256(f"{self.config.name}:{sample['snapshot_id']}".encode()).digest()
            sample["random_tie"] = int.from_bytes(digest[:8]) / (2**64 - 1)
        return sorted(
            samples,
            key=lambda sample: (not sample["conflict"], sample["coverage_score"], sample["random_tie"]),
        )

    @staticmethod
    def _matches(sample: dict[str, Any], filters: dict[str, str]) -> bool:
        for key in ("kit_id", "role", "concept_id", "machine_decision"):
            if filters.get(key) and sample[key] != filters[key]:
                return False
        status = filters.get("status")
        if status == "unreviewed" and sample["review"] is not None:
            return False
        if status in ("accept", "reject", "review") and (
            sample["review"] is None or sample["review"].outcome != status
        ):
            return False
        if filters.get("conflict") == "yes" and not sample["conflict"]:
            return False
        if filters.get("introduced_by") and (
            sample["review"] is None or sample["review"].introduced_by != filters["introduced_by"]
        ):
            return False
        return True

    def sample(self, sample_id: str) -> dict[str, Any] | None:
        return next((sample for sample in self.samples() if sample["sample_id"] == sample_id), None)

    def submit(self, submission: ReviewSubmission) -> ReviewEvent:
        sample = self.sample(submission.sample_id)
        if sample is None or sample["snapshot_id"] != submission.snapshot_id:
            raise ValueError("The reviewed sample snapshot is stale or unknown")
        if submission.outcome == "reject" and submission.issue_code not in self.config.review.issues:
            raise ValueError("The rejection issue is not configured for this run")
        prior = next(
            (
                event
                for event in reversed(self.store.active_reviews())
                if self.store.asset_id(event) == sample["sample_id"]
            ),
            None,
        )
        event = ReviewEvent(
            reviewer=self.reviewer,
            run=self.config.name,
            sample_id=sample["sample_id"],
            snapshot_id=sample["snapshot_id"],
            outputs=sample["outputs"],
            outcome=submission.outcome,
            issue_code=submission.issue_code,
            introduced_by=submission.introduced_by,
            notes=submission.notes,
            supersedes=prior.event_id if prior else None,
        )
        self.store.append_event(event)
        return event

    def undo(self, event_id: str) -> ReviewEvent:
        target = next((event for event in self.store.active_reviews() if event.event_id == event_id), None)
        if target is None:
            raise ValueError("Review event is unknown or already superseded")
        event = ReviewEvent(
            event_type="undo",
            reviewer=self.reviewer,
            run=self.config.name,
            sample_id=target.sample_id,
            snapshot_id=target.snapshot_id,
            outputs=target.outputs,
            supersedes=target.event_id,
        )
        self.store.append_event(event)
        return event

    def media_path(self, record_id: str, level: str | None = None) -> Path:
        record = self.store.record(record_id)
        if record is None:
            raise KeyError(record_id)
        value = record.get("artifact")
        if level is not None:
            lod_level = next((item for item in record.get("levels", []) if item.get("name") == level), None)
            if lod_level is None:
                raise KeyError(f"{record_id}:{level}")
            value = lod_level.get("preview")
        if not value:
            raise KeyError(record_id)
        path = resolve_path(value, self.config.data_dir).resolve()
        data_root = self.config.data_dir.resolve()
        if not path.is_relative_to(data_root) or not path.is_file():
            raise KeyError(record_id)
        return path

    def pyramid_level(self, record_id: str, width: int) -> dict[str, Any]:
        record = self.store.record(record_id)
        if record is None or record.get("stage") != "pyramid":
            raise KeyError(record_id)
        level = next((item for item in record.get("pyramid_levels", []) if item.get("width") == width), None)
        if level is None:
            raise KeyError(f"{record_id}:{width}")
        archive_path = resolve_path(record["artifact"], self.config.data_dir).resolve()
        data_root = self.config.data_dir.resolve()
        if not archive_path.is_relative_to(data_root) or not archive_path.is_file():
            raise KeyError(record_id)
        data = self.pyramid_cache.level(record, archive_path, width)
        source_lods = level.get("source_lods")
        if source_lods is None:
            source_lods = [{"name": level["source_lod"], "weight": 1.0}]
        return {
            "width": width,
            "rows": level["rows"],
            "source_lod": level["source_lod"],
            "source_lods": source_lods,
            **ansi_to_runs(data, width=width, rows=level["rows"]),
        }

    def metrics(self) -> dict[str, Any]:
        samples = self.samples()
        events = self.store.active_reviews()
        events_by_snapshot = {event.snapshot_id: event for event in events}
        outcomes = Counter(event.outcome for event in events)
        matrix = Counter()
        causes = Counter()
        versions: dict[str, Counter[str]] = defaultdict(Counter)
        for event in events:
            if event.outcome == "review":
                continue
            verification = self.store.record(event.outputs.get("verify", ""))
            decision = verification.get("verification", {}).get("decision") if verification else "missing"
            if event.outcome == "accept" and decision == "accept":
                matrix["correct_accept"] += 1
            elif event.outcome == "reject" and decision == "reject":
                matrix["correct_reject"] += 1
            elif event.outcome == "reject" and decision == "accept":
                matrix["unsafe_accept"] += 1
            elif event.outcome == "accept" and decision in ("reject", "review"):
                matrix["wasted_reject"] += 1
            else:
                matrix["missing_or_review"] += 1
            if event.introduced_by:
                causes[event.introduced_by] += 1
            version = verification.get("llm_prompt_version", "missing") if verification else "missing"
            versions[version][f"human_{event.outcome}"] += 1
            versions[version][f"machine_{decision}"] += 1
        coverage = Counter()
        current_reviewed = 0
        for sample in samples:
            event = events_by_snapshot.get(sample["snapshot_id"])
            if event is not None:
                current_reviewed += 1
                coverage[sample["kit_id"]] += 1
        return {
            "total": len(samples),
            "reviewed": current_reviewed,
            "interactions": len(events),
            "definitive": outcomes["accept"] + outcomes["reject"],
            "outcomes": dict(outcomes),
            "matrix": dict(matrix),
            "causes": dict(causes),
            "coverage_by_kit": dict(coverage),
            "totals_by_kit": dict(Counter(sample["kit_id"] for sample in samples)),
            "versions": {key: dict(value) for key, value in versions.items()},
            "manifest_counts": self.store.manifest_counts(),
            "issues": {
                code: [settings.label, settings.default_stage] for code, settings in self.config.review.issues.items()
            },
        }
