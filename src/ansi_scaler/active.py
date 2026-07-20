from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ansi_scaler.config import RunConfig
from ansi_scaler.identity import stable_id
from ansi_scaler.manifests import read_jsonl


def _records(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def _expected_children(
    parents: Iterable[dict[str, Any]], records: Iterable[dict[str, Any]], identity: Any
) -> list[dict[str, Any]]:
    by_id = {record["id"]: record for record in records}
    return [record for parent in parents if (record := by_id.get(identity(parent))) is not None]


def active_rasters(config: RunConfig) -> list[dict[str, Any]]:
    prompts_path = config.manifest_dir / "prompts.jsonl"
    rasters = _records(config.manifest_dir / "rasters.jsonl")
    if not prompts_path.exists():
        return rasters
    settings = config.sana.model_dump(mode="json")
    return _expected_children(
        _records(prompts_path), rasters, lambda parent: stable_id("sana-v1", parent["id"], settings)
    )


def active_backgrounds(config: RunConfig) -> list[dict[str, Any]]:
    backgrounds = _records(config.manifest_dir / "backgrounds.jsonl")
    if not (config.manifest_dir / "prompts.jsonl").exists():
        return backgrounds
    settings = config.background.model_dump(mode="json")
    return _expected_children(
        active_rasters(config),
        backgrounds,
        lambda parent: stable_id("background-v1", parent["id"], settings),
    )


def active_lods(config: RunConfig) -> list[dict[str, Any]]:
    lods = _records(config.manifest_dir / "lods.jsonl")
    if not (config.manifest_dir / "prompts.jsonl").exists():
        return lods
    settings = config.lod.model_dump(mode="json")
    return _expected_children(
        active_backgrounds(config), lods, lambda parent: stable_id("lod-v1", parent["id"], settings)
    )


def active_classifications(config: RunConfig) -> list[dict[str, Any]]:
    classifications = _records(config.manifest_dir / "classifications.jsonl")
    if not (config.manifest_dir / "prompts.jsonl").exists():
        return classifications
    identity = {
        "model": config.vlm.model,
        "prompt_version": config.vlm.prompt_version,
        "temperature": config.vlm.temperature,
        "num_predict": config.vlm.num_predict,
    }
    return _expected_children(
        active_backgrounds(config),
        classifications,
        lambda parent: stable_id("vlm-classify-v1", parent["id"], identity),
    )
