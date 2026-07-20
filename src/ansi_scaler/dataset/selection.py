from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ansi_scaler.config import RunConfig
from ansi_scaler.identity import stable_id
from ansi_scaler.manifests import read_jsonl, resolve_path
from ansi_scaler.review.models import ReviewEvent
from ansi_scaler.stages.classify import OllamaClassifier
from ansi_scaler.stages.verify import OllamaVerifier


VISUAL_STAGES = ("prompt", "generate", "background", "lod", "pyramid")


def _records(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def _children(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        result[record.get("parent_id", "")].append(record)
    return result


def _active_reviews(path: Path) -> dict[str, ReviewEvent]:
    if not path.exists():
        return {}
    events = [ReviewEvent.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    superseded = {event.supersedes for event in events if event.supersedes}
    result: dict[str, ReviewEvent] = {}
    for event in events:
        # Stage-scoped v3 reviews calibrate the pipeline and do not yet control corpus inclusion.
        if event.schema_version < 3 and event.event_type == "set" and event.event_id not in superseded:
            result[event.outputs.get("prompt", event.sample_id)] = event
    return result


def select_assets(config: RunConfig, pyramid_format: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    manifest = config.manifest_dir
    prompts = _records(manifest / "prompts.jsonl")
    generations = _children(_records(manifest / "rasters.jsonl"))
    backgrounds = _children(_records(manifest / "backgrounds.jsonl"))
    lods = _children(_records(manifest / "lods.jsonl"))
    pyramids = _children(
        [r for r in _records(manifest / "pyramids.jsonl") if r.get("pyramid_format") == pyramid_format]
    )
    classifications = _children(_records(manifest / "classifications.jsonl"))
    verifications = _children(_records(manifest / "verifications.jsonl"))
    reviews = _active_reviews(config.run_dir / "reviews" / "annotations.jsonl")
    selected: list[dict[str, Any]] = []
    for prompt in prompts[:limit]:
        chain: dict[str, dict[str, Any]] = {"prompt": prompt}
        parent = prompt["id"]
        for stage, children in (
            ("generate", generations),
            ("background", backgrounds),
            ("lod", lods),
            ("pyramid", pyramids),
        ):
            candidates = children.get(parent, [])
            if not candidates:
                break
            chain[stage] = candidates[-1]
            parent = candidates[-1]["id"]
        background = chain.get("background")
        classifier = classifications.get(background.get("id", ""), []) if background else []
        expected_classifier = OllamaClassifier(config).output_id(background) if background else None
        current_classifier = next((record for record in classifier if record["id"] == expected_classifier), None)
        if current_classifier:
            chain["classify"] = current_classifier
            verifier = verifications.get(current_classifier["id"], [])
            expected_verifier = OllamaVerifier(config).output_id(current_classifier)
            current_verifier = next((record for record in verifier if record["id"] == expected_verifier), None)
            if current_verifier:
                chain["verify"] = current_verifier
        outputs = {stage: record["id"] for stage, record in chain.items()}
        review = reviews.get(prompt["id"])
        applicable = review is not None and all(
            review.outputs.get(stage) == outputs.get(stage) for stage in VISUAL_STAGES
        )
        complete = "pyramid" in chain and resolve_path(chain["pyramid"]["artifact"], config.data_dir).is_file()
        verifier_decision = chain.get("verify", {}).get("verification", {}).get("decision")
        if applicable:
            include = review.outcome == "accept"
            reason = f"human_{review.outcome}"
        elif complete and verifier_decision == "accept":
            include, reason = True, "verifier_accept"
        elif not complete:
            include, reason = False, "incomplete_visual_chain"
        elif verifier_decision is None:
            include, reason = False, "verifier_missing"
        else:
            include, reason = False, f"verifier_{verifier_decision}"
        family = prompt.get("prompt_family_id", prompt.get("prompt_id", prompt["id"]))
        selected.append(
            {
                "asset_id": prompt["id"],
                "prompt_family_id": family,
                "outputs": outputs,
                "pyramid_artifact": chain.get("pyramid", {}).get("artifact"),
                "pyramid_sha256": chain.get("pyramid", {}).get("archive_sha256"),
                "include": include,
                "selection_reason": reason,
                "review_event_id": review.event_id if review else None,
                "review_applicable": applicable,
                "verifier_decision": verifier_decision,
                "prompt": prompt,
                "pyramid": chain.get("pyramid"),
            }
        )
    return selected


def assign_split(family_id: str, seed: int, splits: Any) -> str:
    value = int(stable_id("dataset-split-v1", seed, family_id)[:16], 16) / 2**64
    if value < splits.train:
        return "train"
    if value < splits.train + splits.validation:
        return "validation"
    return "test"


def public_selection(record: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in record.items() if key not in ("prompt", "pyramid")}
    prompt = record.get("prompt")
    if prompt is not None:
        public["prompt_metadata"] = prompt
    return public
