from __future__ import annotations

from pathlib import Path

import numpy as np

from ansi_scaler.dataset.reader import CompiledDataset
from ansi_scaler.identity import sha256_file


def validate_dataset(path: Path) -> dict[str, int]:
    dataset = CompiledDataset.open(path)
    metadata = dataset.metadata
    if metadata["vocabulary_sha256"] != sha256_file(path / "vocabulary.json"):
        raise ValueError("Vocabulary checksum mismatch")
    if metadata["selection_sha256"] != sha256_file(path / "selection.jsonl"):
        raise ValueError("Selection checksum mismatch")
    if metadata["index_sha256"] != sha256_file(path / "index.jsonl"):
        raise ValueError("Index checksum mismatch")
    for shard in metadata["shards"]:
        if sha256_file(path / shard["path"]) != shard["sha256"]:
            raise ValueError(f"Shard checksum mismatch: {shard['path']}")
    assets = levels = cells = 0
    family_splits: dict[str, str] = {}
    for asset in dataset.assets():
        assets += 1
        family = asset.record["prompt_family_id"]
        previous = family_splits.setdefault(family, asset.record["split"])
        if previous != asset.record["split"]:
            raise ValueError(f"Prompt family crosses splits: {family}")
        for level in asset.levels():
            levels += 1
            count = level.width * level.rows
            cells += count
            if len(level.glyph_ids) != count:
                raise ValueError(f"Cell count mismatch for {asset.record['asset_id']} width {level.width}")
            if level.glyph_ids.dtype != np.uint16:
                raise ValueError("Glyph IDs are not uint16")
            if level.foreground_rgb.dtype != np.uint8 or level.background_rgb.dtype != np.uint8:
                raise ValueError("Cell colours are not uint8")
            if level.background_present.dtype != np.bool_:
                raise ValueError("Background presence is not boolean")
            if not np.isclose(level.lod_weights.sum(), 1.0):
                raise ValueError("LOD weights do not sum to one")
    if assets != metadata["asset_count"]:
        raise ValueError("Asset count does not match metadata")
    return {"assets": assets, "levels": levels, "cells": cells}
