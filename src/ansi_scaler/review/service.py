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
from ansi_scaler.stages.background import BackgroundProcessor
from ansi_scaler.stages.classify import OllamaClassifier
from ansi_scaler.stages.generate import SanaGenerator
from ansi_scaler.stages.lod import LodGenerator
from ansi_scaler.stages.pyramid import pyramid_id
from ansi_scaler.stages.verify import OllamaVerifier


REVIEW_ORDER = ("generate", "background", "classify", "verify", "lod", "pyramid")
REVIEW_LINEAGES = {
    "generate": ("prompt", "generate"),
    "background": ("prompt", "generate", "background"),
    "lod": ("prompt", "generate", "background", "lod"),
    "classify": ("prompt", "generate", "background", "classify"),
    "verify": ("prompt", "generate", "background", "classify", "verify"),
    "pyramid": ("prompt", "generate", "background", "lod", "pyramid"),
}


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
        active_reviews: dict[str, list[ReviewEvent]] = defaultdict(list)
        for event in self.store.active_reviews():
            active_reviews[self.store.asset_id(event)].append(event)
        all_reviews: dict[str, list[ReviewEvent]] = defaultdict(list)
        for event in self.store.review_events():
            if event.event_type == "set":
                all_reviews[self.store.asset_id(event)].append(event)
        errors = {record.get("output_id"): record for record in self.store.errors() if record.get("output_id")}

        generator = SanaGenerator(self.config)
        backgrounder = BackgroundProcessor(self.config)
        lod_generator = LodGenerator(self.config)
        classifier = OllamaClassifier(self.config)
        verifier = OllamaVerifier(self.config)

        samples = []
        for prompt in by_stage["prompts"]:
            expected: dict[str, str] = {"prompt": prompt["id"], "generate": generator.output_id(prompt)}
            raster = next((item for item in children[(prompt["id"], "generate")] if item["id"] == expected["generate"]), None)
            cutout = lod = pyramid = classification = verification = None
            if raster is not None:
                expected["background"] = backgrounder.output_id(raster)
                cutout = next((item for item in children[(raster["id"], "background")] if item["id"] == expected["background"]), None)
            if cutout is not None:
                expected["classify"] = classifier.output_id(cutout)
                expected["lod"] = lod_generator.output_id(cutout)
                classification = next((item for item in children[(cutout["id"], "classify")] if item["id"] == expected["classify"]), None)
                lod = next((item for item in children[(cutout["id"], "lod")] if item["id"] == expected["lod"]), None)
            if classification is not None:
                expected["verify"] = verifier.output_id(classification)
                verification = next((item for item in children[(classification["id"], "verify")] if item["id"] == expected["verify"]), None)
            if lod is not None:
                expected["pyramid"] = pyramid_id(lod, self.config)
                pyramid = next((item for item in children[(lod["id"], "pyramid")] if item["id"] == expected["pyramid"]), None)
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
            stage_targets = {stage: output_id for stage, output_id in expected.items() if stage != "prompt"}
            stage_errors = {
                stage: errors[output_id]
                for stage, output_id in stage_targets.items()
                if output_id in errors and stage not in outputs
            }
            snapshot_id = stable_id("review-snapshot-v1", prompt["id"], stage_targets)
            asset_id = prompt["id"]
            machine_decision = verification.get("verification", {}).get("decision") if verification else "missing"
            current_reviews: dict[str, ReviewEvent] = {}
            stale_reviews: dict[str, ReviewEvent] = {}
            legacy_review: ReviewEvent | None = None
            for event in active_reviews.get(asset_id, []):
                if event.schema_version < 3:
                    if event.snapshot_id == snapshot_id:
                        legacy_review = event
                    continue
                if event.target_stage:
                    target = current_reviews if self._review_applies(event, outputs) else stale_reviews
                    target[event.target_stage] = event
            sample = {
                "sample_id": asset_id,
                "snapshot_id": snapshot_id,
                "outputs": outputs,
                "stage_targets": stage_targets,
                "stage_errors": stage_errors,
                "records": records,
                "prompt": prompt,
                "raster": raster,
                "cutout": cutout,
                "lod": lod,
                "pyramid": pyramid,
                "classification": classification,
                "verification": verification,
                "machine_decision": machine_decision,
                "review": None,
                "reviews": current_reviews,
                "stale_reviews": stale_reviews,
                "legacy_review": legacy_review,
                "history": all_reviews.get(asset_id, []),
                "kit_id": (raster or prompt).get("location", (raster or prompt).get("kit_id", "unknown")),
                "role": (raster or prompt).get("theme", (raster or prompt).get("role", "unknown")),
                "concept_id": (raster or prompt).get("specification_id", (raster or prompt).get("concept_id", "unknown")),
                "concept_name": (raster or prompt).get("label", (raster or prompt).get("concept_name", (raster or prompt).get("concept_id", "Unknown"))),
            }
            focus_stage, complete = self._review_state(sample)
            sample["focus_stage"] = focus_stage
            sample["focus_output_id"] = stage_targets.get(focus_stage) if focus_stage else None
            sample["review_complete"] = complete
            sample["review"] = current_reviews.get(focus_stage) if focus_stage else legacy_review
            sample["conflict"] = self._has_conflict(sample)
            sample["stage_tabs"] = self._stage_tabs(sample)
            samples.append(sample)
        self._sample_cache = (self.store.revision, samples)
        return samples

    @staticmethod
    def _review_applies(event: ReviewEvent, outputs: dict[str, str]) -> bool:
        return bool(event.lineage) and all(
            outputs.get(stage) == output_id for stage, output_id in event.lineage.items()
        )

    @staticmethod
    def _available_stages(sample: dict[str, Any]) -> list[str]:
        return [stage for stage in REVIEW_ORDER if stage in sample["outputs"] or stage in sample["stage_errors"]]

    def _review_state(self, sample: dict[str, Any]) -> tuple[str | None, bool]:
        available = self._available_stages(sample)
        if not available:
            return None, True
        reviews: dict[str, ReviewEvent] = sample["reviews"]
        terminal = [
            stage for stage in available if reviews.get(stage) and reviews[stage].outcome in ("reject", "unsure")
        ]
        if terminal:
            return terminal[0], True
        stale = [stage for stage in available if stage in sample["stale_reviews"]]
        if stale:
            return stale[0], False
        failures = [stage for stage in available if stage in sample["stage_errors"]]
        if sample["verification"] and sample["verification"].get("verification", {}).get("decision") in ("reject", "review"):
            failures.append("verify")
        failures = sorted(set(failures), key=REVIEW_ORDER.index)
        for failed_stage in failures:
            if not (reviews.get(failed_stage) and reviews[failed_stage].outcome == "accept"):
                return failed_stage, False
        accepted = [stage for stage in available if reviews.get(stage) and reviews[stage].outcome == "accept"]
        if accepted:
            latest = max(accepted, key=REVIEW_ORDER.index)
            later = [stage for stage in available if REVIEW_ORDER.index(stage) > REVIEW_ORDER.index(latest)]
            if later:
                return later[0], False
            return latest, True
        return available[-1], False

    @staticmethod
    def _stage_tabs(sample: dict[str, Any]) -> list[dict[str, Any]]:
        labels = {
            "generate": "Generate",
            "background": "Background",
            "classify": "Classify",
            "verify": "Verify",
            "lod": "LOD",
            "pyramid": "ANSI",
        }
        tabs = []
        for stage in REVIEW_ORDER:
            review = sample["reviews"].get(stage)
            status = "pending"
            if stage in sample["outputs"] or stage in sample["stage_errors"]:
                status = "ready"
            if stage in sample["stage_errors"]:
                status = "failed"
            if stage == "verify" and sample["verification"]:
                decision = sample["verification"].get("verification", {}).get("decision")
                if decision in ("reject", "review"):
                    status = "failed" if decision == "reject" else "unsure"
            if review:
                status = {"reject": "failed", "unsure": "unsure", "accept": "accepted"}[review.outcome]
            tabs.append(
                {
                    "stage": stage,
                    "label": labels[stage],
                    "status": status,
                    "selected": stage == sample["focus_stage"],
                    "disabled": stage not in sample["outputs"] and stage not in sample["stage_errors"],
                    "output_id": sample["stage_targets"].get(stage),
                    "event_id": review.event_id if review else "",
                    "error": sample["stage_errors"].get(stage),
                }
            )
        return tabs

    @staticmethod
    def _has_conflict(sample: dict[str, Any]) -> bool:
        decision = sample["machine_decision"]
        reviews = sample["reviews"].values()
        return bool(sample["stale_reviews"]) or any(
            (event.outcome == "accept" and decision in ("reject", "review"))
            or (event.outcome == "reject" and decision == "accept")
            for event in reviews
        )

    def queue(self, filters: dict[str, str] | None = None, *, include_reviewed: bool = False) -> list[dict[str, Any]]:
        filters = filters or {}
        samples = [sample for sample in self.samples() if self._matches(sample, filters)]
        if not include_reviewed:
            samples = [sample for sample in samples if not sample["review_complete"]]
        all_samples = self.samples()
        definitive = [sample for sample in all_samples if sample["review_complete"] and sample["review"]]
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
        if status == "unreviewed" and sample["review_complete"]:
            return False
        normalized_status = "unsure" if status == "review" else status
        if normalized_status in ("accept", "reject", "unsure") and (
            sample["review"] is None or sample["review"].outcome != normalized_status
        ):
            return False
        if filters.get("conflict") == "yes" and not sample["conflict"]:
            return False
        return True

    def sample(self, sample_id: str) -> dict[str, Any] | None:
        return next((sample for sample in self.samples() if sample["sample_id"] == sample_id), None)

    def submit(self, submission: ReviewSubmission) -> ReviewEvent:
        sample = self.sample(submission.sample_id)
        if sample is None:
            raise ValueError("The reviewed sample is unknown")
        if submission.target_stage not in REVIEW_LINEAGES:
            raise ValueError("The reviewed stage is unknown")
        if submission.target_stage not in sample["outputs"] and submission.target_stage not in sample["stage_errors"]:
            raise ValueError("The reviewed stage has not produced an output or error")
        if sample["stage_targets"].get(submission.target_stage) != submission.target_output_id:
            raise ValueError("The reviewed stage output is stale or unknown")
        lineage = {
            stage: sample["stage_targets"].get(stage, sample["outputs"].get(stage))
            for stage in REVIEW_LINEAGES[submission.target_stage]
            if stage in sample["stage_targets"] or stage in sample["outputs"]
        }
        if lineage.get(submission.target_stage) != submission.target_output_id:
            raise ValueError("The reviewed stage does not have a complete lineage")
        prior = next(
            (
                event
                for event in reversed(self.store.active_reviews())
                if self.store.asset_id(event) == sample["sample_id"]
                and event.schema_version == 3
                and event.target_stage == submission.target_stage
            ),
            None,
        )
        event = ReviewEvent(
            reviewer=self.reviewer,
            run=self.config.name,
            sample_id=sample["sample_id"],
            snapshot_id=sample["snapshot_id"],
            outputs=sample["outputs"],
            target_stage=submission.target_stage,
            target_output_id=submission.target_output_id,
            lineage=lineage,
            outcome=submission.outcome,
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
            schema_version=target.schema_version,
            event_type="undo",
            reviewer=self.reviewer,
            run=self.config.name,
            sample_id=target.sample_id,
            snapshot_id=target.snapshot_id,
            outputs=target.outputs,
            target_stage=target.target_stage,
            target_output_id=target.target_output_id,
            lineage=target.lineage,
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
        current_event_ids = {
            event.event_id
            for sample in samples
            for event in [*sample["reviews"].values(), *([sample["legacy_review"]] if sample["legacy_review"] else [])]
        }
        events = [event for event in self.store.active_reviews() if event.event_id in current_event_ids]
        outcomes = Counter(event.outcome for event in events)
        matrix = Counter()
        causes = Counter()
        stage_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
        versions: dict[str, Counter[str]] = defaultdict(Counter)
        for event in events:
            verification = self.store.record(event.outputs.get("verify", ""))
            decision = verification.get("verification", {}).get("decision") if verification else "missing"
            stage = event.target_stage or event.introduced_by or "legacy"
            causes[stage] += 1
            stage_outcomes[stage][event.outcome or "unknown"] += 1
            if stage == "classify":
                classification = self.store.record(event.target_output_id or event.outputs.get("classify", ""))
                version = classification.get("vlm_prompt_version", "missing") if classification else "missing"
            else:
                version = verification.get("llm_prompt_version", "missing") if verification else "missing"
            versions[version][f"human_{event.outcome}"] += 1
            versions[version][f"machine_{decision}"] += 1
            if event.outcome in ("review", "unsure"):
                continue
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
        coverage = Counter()
        completed = [sample for sample in samples if sample["review_complete"]]
        for sample in completed:
            if sample["review"] is not None:
                coverage[sample["kit_id"]] += 1
        return {
            "total": len(samples),
            "reviewed": len(completed),
            "interactions": len(events),
            "definitive": outcomes["accept"] + outcomes["reject"],
            "outcomes": dict(outcomes),
            "matrix": dict(matrix),
            "causes": dict(causes),
            "stage_outcomes": {key: dict(value) for key, value in stage_outcomes.items()},
            "coverage_by_kit": dict(coverage),
            "totals_by_kit": dict(Counter(sample["kit_id"] for sample in samples)),
            "versions": {key: dict(value) for key, value in versions.items()},
            "manifest_counts": self.store.manifest_counts(),
        }
