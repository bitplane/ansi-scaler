from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ansi_scaler.config import RunConfig
from ansi_scaler.identity import sha256_file
from ansi_scaler.manifests import read_jsonl, resolve_path, write_jsonl
from ansi_scaler.review.models import ReviewEvent
from ansi_scaler.stages.classify import OllamaClassifier
from ansi_scaler.stages.generate import SanaGenerator
from ansi_scaler.stages.lod import LodGenerator
from ansi_scaler.stages.pyramid import pyramid_id
from ansi_scaler.stages.background import BackgroundProcessor
from ansi_scaler.stages.verify import OllamaVerifier


KNOWN_STAGES = {"prompts", "generate", "background", "lod", "pyramid", "classify", "verify"}
MANAGED_ARTIFACT_ROOTS = ("rasters", "backgrounds", "lod", "pyramids")
VISUAL_DEPTH = {"generate": 1, "background": 2, "lod": 3, "pyramid": 4}
SEMANTIC_DEPTH = {"generate": 1, "background": 2, "classify": 3, "verify": 4}


@dataclass
class ManifestChange:
    path: Path
    before: list[dict[str, Any]]
    after: list[dict[str, Any]]


@dataclass
class GcPlan:
    config_path: Path
    config: RunConfig
    changes: list[ManifestChange]
    delete_paths: list[Path]
    orphan_paths: set[Path]
    missing_paths: set[Path]
    missing_by_run: Counter[str]
    fingerprints: dict[Path, str]
    retained_reasons: Counter[str] = field(default_factory=Counter)
    removed_by_stage: Counter[str] = field(default_factory=Counter)
    deleted_by_section: Counter[str] = field(default_factory=Counter)

    @property
    def reclaimable_bytes(self) -> int:
        return sum(disk_bytes(path) for path in self.delete_paths)

    @property
    def removed_records(self) -> int:
        return sum(len(change.before) - len(change.after) for change in self.changes)


def disk_bytes(path: Path) -> int:
    stat = path.stat()
    return stat.st_blocks * 512


def _record_artifacts(record: dict[str, Any]) -> set[str]:
    stage = record.get("stage")
    if stage in {"generate", "background", "pyramid"} and record.get("artifact"):
        return {record["artifact"]}
    if stage == "lod":
        return {level[key] for level in record.get("levels", []) for key in ("svg", "preview") if level.get(key)}
    return set()


def _all_artifact_strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if value.startswith("artifacts/") else set()
    if isinstance(value, dict):
        paths: set[str] = set()
        for item in value.values():
            paths.update(_all_artifact_strings(item))
        return paths
    if isinstance(value, list):
        paths = set()
        for item in value:
            paths.update(_all_artifact_strings(item))
        return paths
    return set()


def _fingerprint(path: Path) -> str:
    return sha256_file(path) if path.exists() else "missing"


def _manifest_paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("runs/*/manifests/*.jsonl"))


def _active_reviews(path: Path) -> list[ReviewEvent]:
    if not path.exists():
        return []
    events = [ReviewEvent.model_validate(record) for record in read_jsonl(path)]
    superseded = {event.supersedes for event in events if event.supersedes}
    active = [event for event in events if event.event_type == "set" and event.event_id not in superseded]
    by_target: dict[tuple[str, str | None], ReviewEvent] = {}
    for event in active:
        asset = event.outputs.get("prompt", event.sample_id)
        key = (asset, event.target_stage if event.schema_version == 3 else None)
        by_target[key] = event
    return list(by_target.values())


def _root_prompt(record_id: str, records: dict[str, dict[str, Any]]) -> str | None:
    seen: set[str] = set()
    current = records.get(record_id)
    while current is not None and current["id"] not in seen:
        seen.add(current["id"])
        if current.get("stage") == "prompts":
            return current["id"]
        current = records.get(current.get("parent_id"))
    return None


def _current_ids(prompt: dict[str, Any], config: RunConfig) -> dict[str, str]:
    raster = SanaGenerator(config).output_id(prompt)
    cutout = BackgroundProcessor(config).output_id({"id": raster})
    lod = LodGenerator(config).output_id({"id": cutout})
    classification = OllamaClassifier(config).output_id({"id": cutout})
    return {
        "prompts": prompt["id"],
        "generate": raster,
        "background": cutout,
        "lod": lod,
        "pyramid": pyramid_id({"id": lod}, config),
        "classify": classification,
        "verify": OllamaVerifier(config).output_id({"id": classification}),
    }


def _latest_for_prompt(
    records_in_order: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    prompt_id: str,
    depths: dict[str, int],
) -> dict[str, Any] | None:
    candidates = [
        (depths.get(record.get("stage"), 0), index, record)
        for index, record in enumerate(records_in_order)
        if record.get("stage") in depths and _root_prompt(record["id"], records) == prompt_id
    ]
    return max(candidates, default=(0, 0, None), key=lambda item: (item[0], item[1]))[2]


def _mark_with_ancestors(root_id: str, records: dict[str, dict[str, Any]], retained: set[str]) -> None:
    current = records.get(root_id)
    while current is not None and current["id"] not in retained:
        retained.add(current["id"])
        current = records.get(current.get("parent_id"))


def _section(path: Path, artifact_root: Path) -> str:
    relative = path.relative_to(artifact_root)
    return relative.parts[0] if relative.parts else "unknown"


def _valid_records(records: dict[str, dict[str, Any]], config: RunConfig) -> tuple[dict[str, dict[str, Any]], set[str]]:
    invalid = {
        record_id
        for record_id, record in records.items()
        if record.get("stage") in KNOWN_STAGES
        and any(not resolve_path(value, config.data_dir).is_file() for value in _record_artifacts(record))
    }
    changed = True
    while changed:
        changed = False
        for record_id, record in records.items():
            if record_id in invalid or record.get("stage") in {"prompts"} or record.get("stage") not in KNOWN_STAGES:
                continue
            parent_id = record.get("parent_id")
            if parent_id not in records or parent_id in invalid:
                invalid.add(record_id)
                changed = True
    return {record_id: record for record_id, record in records.items() if record_id not in invalid}, invalid


def build_gc_plan(config: RunConfig, config_path: Path) -> GcPlan:
    target_paths = sorted(config.manifest_dir.glob("*.jsonl"))
    success_paths = [path for path in target_paths if not path.name.endswith(".errors.jsonl")]
    error_paths = [path for path in target_paths if path.name.endswith(".errors.jsonl")]
    prompt_path = config.manifest_dir / "prompts.jsonl"
    if not prompt_path.is_file():
        raise ValueError(f"Refusing to collect a run without its prompt manifest: {prompt_path}")
    records_in_order: list[dict[str, Any]] = []
    manifest_records: dict[Path, list[dict[str, Any]]] = {}
    records: dict[str, dict[str, Any]] = {}
    for path in success_paths:
        loaded = list(read_jsonl(path))
        manifest_records[path] = loaded
        for record in loaded:
            if not record.get("id") or not record.get("stage"):
                raise ValueError(f"Manifest record lacks id or stage: {path}")
            previous = records.get(record["id"])
            if previous is not None and previous != record:
                raise ValueError(f"Conflicting records share id {record['id']}")
            records[record["id"]] = record
            records_in_order.append(record)

    all_records = records
    records, invalid_ids = _valid_records(all_records, config)
    prompts = [record for record in records_in_order if record.get("stage") == "prompts" and record["id"] in records]
    retained: set[str] = set()
    expected_ids: set[str] = set()
    reasons: Counter[str] = Counter()
    for prompt in prompts:
        ids = _current_ids(prompt, config)
        expected_ids.update(ids.values())
        retained.add(prompt["id"])
        for branch in (
            ("generate", "background", "lod", "pyramid"),
            ("generate", "background", "classify", "verify"),
        ):
            current_terminal = False
            for stage in branch:
                record_id = ids[stage]
                if record_id not in records:
                    break
                _mark_with_ancestors(record_id, records, retained)
                if stage == branch[-1]:
                    current_terminal = True
            if current_terminal:
                reasons["current complete"] += 1
                continue
            depths = VISUAL_DEPTH if branch[-1] == "pyramid" else SEMANTIC_DEPTH
            fallback = _latest_for_prompt(records_in_order, records, prompt["id"], depths)
            if fallback is not None:
                _mark_with_ancestors(fallback["id"], records, retained)
                reasons["handover fallback"] += 1
            reasons["current partial"] += 1

    annotations = config.run_dir / "reviews" / "annotations.jsonl"
    for event in _active_reviews(annotations):
        references = event.lineage if event.schema_version == 3 else event.outputs
        for record_id in references.values():
            _mark_with_ancestors(record_id, records, retained)
        reasons["active review"] += 1

    for record in records_in_order:
        if record.get("stage") not in KNOWN_STAGES:
            _mark_with_ancestors(record["id"], records, retained)
            reasons["unknown record"] += 1

    changes: list[ManifestChange] = []
    removed_by_stage: Counter[str] = Counter()
    retained_records: list[dict[str, Any]] = []
    for path, before in manifest_records.items():
        after = [record for record in before if record["id"] in retained and record["id"] not in invalid_ids]
        retained_records.extend(after)
        removed_by_stage.update(record["stage"] for record in before if record["id"] not in retained)
        changes.append(ManifestChange(path, before, after))

    for path in error_paths:
        before = list(read_jsonl(path))
        after = [
            record
            for record in before
            if not record.get("output_id")
            or (record["output_id"] in expected_ids and record["output_id"] not in records)
        ]
        changes.append(ManifestChange(path, before, after))
        removed_by_stage["errors"] += len(before) - len(after)

    live_strings: set[str] = set()
    for record in retained_records:
        live_strings.update(_record_artifacts(record))
        if record.get("stage") not in KNOWN_STAGES:
            live_strings.update(_all_artifact_strings(record))

    cross_run_refs: dict[Path, set[str]] = defaultdict(set)
    for path in _manifest_paths(config.data_dir):
        if path.parent == config.manifest_dir:
            continue
        run_name = path.parent.parent.name
        for record in read_jsonl(path):
            for value in _all_artifact_strings(record):
                live_strings.add(value)
                cross_run_refs[resolve_path(value, config.data_dir).resolve()].add(run_name)
    live_paths = {resolve_path(value, config.data_dir).resolve() for value in live_strings}

    artifact_root = config.artifact_dir.resolve()
    escaped = sorted(path for path in live_paths if not path.is_relative_to(artifact_root))
    if escaped:
        raise ValueError(f"Manifest artifact path escapes the managed artifact root: {escaped[0]}")
    for root_name in MANAGED_ARTIFACT_ROOTS:
        symlink = next((path for path in (artifact_root / root_name).rglob("*") if path.is_symlink()), None)
        if symlink is not None:
            raise ValueError(f"Managed artifact trees may not contain symlinks: {symlink}")
    managed_files = {
        path.resolve()
        for root_name in MANAGED_ARTIFACT_ROOTS
        for path in (artifact_root / root_name).rglob("*")
        if path.is_file()
    }
    missing_paths = {path for path in live_paths if not path.is_file()}
    missing_by_run: Counter[str] = Counter()
    for path in missing_paths:
        owners = cross_run_refs.get(path, {config.name})
        missing_by_run.update(owners)
    delete_paths = sorted(managed_files - live_paths)

    removed_owned = {
        resolve_path(value, config.data_dir).resolve()
        for change in changes
        for record in change.before
        if record.get("id") and record.get("id") not in retained
        for value in _record_artifacts(record)
    }
    orphan_paths = set(delete_paths) - removed_owned
    deleted_by_section = Counter(_section(path, artifact_root) for path in delete_paths)
    fingerprint_paths = _manifest_paths(config.data_dir) + [annotations, config_path]
    fingerprints = {path.resolve(): _fingerprint(path) for path in fingerprint_paths}
    return GcPlan(
        config_path=config_path,
        config=config,
        changes=changes,
        delete_paths=delete_paths,
        orphan_paths=orphan_paths,
        missing_paths=missing_paths,
        missing_by_run=missing_by_run,
        fingerprints=fingerprints,
        retained_reasons=reasons,
        removed_by_stage=removed_by_stage,
        deleted_by_section=deleted_by_section,
    )


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    raise AssertionError("unreachable")


def plan_report(plan: GcPlan) -> str:
    lines = [f"Corpus GC for run {plan.config.name}", ""]
    lines.append("Records to remove:")
    for stage, count in sorted(plan.removed_by_stage.items()):
        lines.append(f"  {stage:<14} {count:>8}")
    if not plan.removed_by_stage:
        lines.append("  none")
    lines.extend(["", "Artifact files to remove:"])
    artifact_root = plan.config.artifact_dir.resolve()
    for section, count in sorted(plan.deleted_by_section.items()):
        paths = [path for path in plan.delete_paths if _section(path, artifact_root) == section]
        lines.append(f"  {section:<14} {count:>8}  {format_bytes(sum(disk_bytes(path) for path in paths)):>12}")
    if not plan.delete_paths:
        lines.append("  none")
    lines.extend(
        [
            f"  {'true orphans':<14} {len(plan.orphan_paths):>8}  "
            f"{format_bytes(sum(disk_bytes(path) for path in plan.orphan_paths)):>12}",
            "",
            f"Missing references remaining after target cleanup: {len(plan.missing_paths)}",
            "",
            "Retention roots:",
        ]
    )
    for run, count in sorted(plan.missing_by_run.items()):
        lines.append(f"  {run:<20} {count:>8}  run GC with that run's config to prune")
    for reason, count in sorted(plan.retained_reasons.items()):
        lines.append(f"  {reason:<20} {count:>8}")
    lines.extend(
        [
            "",
            f"Total: {plan.removed_records} manifest records, {len(plan.delete_paths)} files, "
            f"{format_bytes(plan.reclaimable_bytes)} reclaimable",
        ]
    )
    return "\n".join(lines)


def _validate_plan(plan: GcPlan) -> None:
    changed = [path for path, digest in plan.fingerprints.items() if _fingerprint(path) != digest]
    if changed:
        raise RuntimeError(f"Corpus changed while GC was planning; aborting before deletion: {changed[0]}")
    artifact_root = plan.config.artifact_dir.resolve()
    for path in plan.delete_paths:
        if not path.is_relative_to(artifact_root) or path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Planned artifact is no longer a safe regular file: {path}")


def apply_gc_plan(plan: GcPlan) -> Path:
    _validate_plan(plan)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    plan_payload = {
        "run": plan.config.name,
        "config": str(plan.config_path),
        "fingerprints": {str(path): digest for path, digest in plan.fingerprints.items()},
        "delete_paths": [str(path) for path in plan.delete_paths],
        "missing_paths": [str(path) for path in sorted(plan.missing_paths)],
        "removed_records": plan.removed_records,
        "manifest_changes": {
            str(change.path): {
                "before": len(change.before),
                "after": len(change.after),
                "removed": [
                    record.get("id") or record.get("output_id") or f"legacy-error:{record.get('parent_id', 'unknown')}"
                    for record in change.before
                    if record not in change.after
                ],
            }
            for change in plan.changes
        },
        "reclaimable_bytes": plan.reclaimable_bytes,
    }
    plan_id = hashlib.sha256(json.dumps(plan_payload, sort_keys=True).encode()).hexdigest()[:12]
    receipt_dir = plan.config.run_dir / "gc" / f"{timestamp}-{plan_id}"
    backup_dir = receipt_dir / "before"
    backup_dir.mkdir(parents=True)
    for change in plan.changes:
        if change.path.exists():
            shutil.copy2(change.path, backup_dir / change.path.name)
    annotations = plan.config.run_dir / "reviews" / "annotations.jsonl"
    if annotations.exists():
        shutil.copy2(annotations, backup_dir / "annotations.jsonl")
    shutil.copy2(plan.config_path, backup_dir / plan.config_path.name)
    (receipt_dir / "plan.json").write_text(json.dumps(plan_payload, indent=2, sort_keys=True) + "\n")

    for change in plan.changes:
        write_jsonl(change.path, change.after)

    # No artifact is touched until every compacted manifest is safely installed.
    deleted_bytes = 0
    for path in plan.delete_paths:
        deleted_bytes += disk_bytes(path)
        path.unlink()
    artifact_root = plan.config.artifact_dir.resolve()
    directories = sorted(
        (path for path in artifact_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in directories:
        try:
            path.rmdir()
        except OSError:
            pass
    review_dir = plan.config.run_dir / "reviews"
    for name in ("index.sqlite3", "index.sqlite3-shm", "index.sqlite3-wal"):
        (review_dir / name).unlink(missing_ok=True)
    receipt = {**plan_payload, "completed_at": datetime.now(UTC).isoformat(), "deleted_bytes": deleted_bytes}
    (receipt_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt_dir
