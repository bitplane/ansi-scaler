from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from safetensors import safe_open


@dataclass(frozen=True)
class CompiledLevel:
    width: int
    rows: int
    glyph_ids: np.ndarray
    foreground_rgb: np.ndarray
    background_rgb: np.ndarray
    background_present: np.ndarray
    lod_weights: np.ndarray


class CompiledAsset:
    def __init__(self, dataset: CompiledDataset, record: dict[str, Any]) -> None:
        self.dataset = dataset
        self.record = record

    def levels(self) -> Iterator[CompiledLevel]:
        shard_path = self.dataset.path / f"shard-{self.record['shard']:05d}.safetensors"
        with safe_open(shard_path, framework="numpy") as handle:
            start = self.record["level_start"]
            end = start + self.record["level_count"]
            offsets = handle.get_tensor("level_cell_offsets")
            widths = handle.get_tensor("level_widths")
            rows = handle.get_tensor("level_rows")
            weights = handle.get_tensor("level_lod_weights")
            for level_index in range(start, end):
                cell_start, cell_end = int(offsets[level_index]), int(offsets[level_index + 1])
                yield CompiledLevel(
                    width=int(widths[level_index]),
                    rows=int(rows[level_index]),
                    glyph_ids=handle.get_slice("glyph_ids")[cell_start:cell_end],
                    foreground_rgb=handle.get_slice("foreground_rgb")[cell_start:cell_end],
                    background_rgb=handle.get_slice("background_rgb")[cell_start:cell_end],
                    background_present=handle.get_slice("background_present")[cell_start:cell_end],
                    lod_weights=weights[level_index],
                )

    def level(self, width: int) -> CompiledLevel:
        for level in self.levels():
            if level.width == width:
                return level
        raise KeyError(f"Asset {self.record['asset_id']} has no width {width}")


class CompiledDataset:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.metadata = json.loads((path / "dataset.json").read_text())
        self.vocabulary = json.loads((path / "vocabulary.json").read_text())
        self._records = [json.loads(line) for line in (path / "index.jsonl").read_text().splitlines() if line.strip()]
        self._by_id = {record["asset_id"]: record for record in self._records}

    @classmethod
    def open(cls, path: Path | str) -> CompiledDataset:
        return cls(Path(path))

    def assets(self, split: str | None = None) -> Iterator[CompiledAsset]:
        return (CompiledAsset(self, record) for record in self._records if split is None or record["split"] == split)

    def asset(self, asset_id: str) -> CompiledAsset:
        return CompiledAsset(self, self._by_id[asset_id])
