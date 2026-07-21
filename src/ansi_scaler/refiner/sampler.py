from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ansi_scaler.dataset.reader import CompiledDataset, CompiledLevel


@dataclass(frozen=True)
class Window:
    source_width: int
    target_width: int
    x: int
    y: int
    kind: str


@dataclass
class AssetPatches:
    asset_id: str
    split: str
    prompt: str
    render_bbox: np.ndarray
    levels: dict[int, CompiledLevel]
    windows: dict[str, list[Window]]


@dataclass(frozen=True)
class Patch:
    asset_id: str
    context_glyphs: np.ndarray
    context_foreground: np.ndarray
    context_background: np.ndarray
    context_background_present: np.ndarray
    target_glyphs: np.ndarray
    target_foreground: np.ndarray
    target_background: np.ndarray
    target_background_present: np.ndarray
    bbox: np.ndarray
    scale: np.ndarray
    source_lod_weights: np.ndarray
    target_lod_weights: np.ndarray


def exact_pairs(levels: dict[int, CompiledLevel]) -> list[tuple[CompiledLevel, CompiledLevel]]:
    result = []
    for source in levels.values():
        if source.width % 2 or source.rows % 2:
            continue
        target = levels.get(source.width * 3 // 2)
        if target is not None and target.rows == source.rows * 3 // 2:
            result.append((source, target))
    return sorted(result, key=lambda pair: pair[0].width)


def _grid(level: CompiledLevel, value: str) -> np.ndarray:
    array = getattr(level, value)
    trailing = array.shape[1:]
    return array.reshape((level.rows, level.width, *trailing))


def aligned_windows(levels: dict[int, CompiledLevel], space_id: int) -> dict[str, list[Window]]:
    result: dict[str, list[Window]] = {"edge": [], "content": [], "empty": []}
    for source, target in exact_pairs(levels):
        if source.width < 8 or source.rows < 4:
            continue
        target_glyphs = _grid(target, "glyph_ids")
        target_background = _grid(target, "background_present")
        for y in range(2, source.rows - 2, 2):
            for x in range(2, source.width - 5, 2):
                tx, ty = x * 3 // 2, y * 3 // 2
                glyphs = target_glyphs[ty : ty + 3, tx : tx + 6]
                backgrounds = target_background[ty : ty + 3, tx : tx + 6]
                visible = np.logical_or(glyphs != space_id, backgrounds)
                count = int(visible.sum())
                kind = "empty" if count == 0 else ("content" if count == 18 else "edge")
                result[kind].append(Window(source.width, target.width, x, y, kind))
    return result


def load_patch_assets(dataset: CompiledDataset, split: str) -> list[AssetPatches]:
    result = []
    space_id = dataset.vocabulary["codepoints"].index(ord(" ")) + 3
    for asset in dataset.assets(split):
        levels = {level.width: level for level in asset.levels()}
        windows = aligned_windows(levels, space_id)
        if not any(windows.values()):
            continue
        metadata = asset.record.get("prompt_metadata", {})
        result.append(
            AssetPatches(
                asset_id=asset.record["asset_id"],
                split=split,
                prompt=metadata.get("prompt", metadata.get("concept_name", "")),
                render_bbox=asset.geometry()["render_bbox"].astype(np.float32),
                levels=levels,
                windows=windows,
            )
        )
    return result


def extract_patch(asset: AssetPatches, window: Window) -> Patch:
    source, target = asset.levels[window.source_width], asset.levels[window.target_width]
    x, y = window.x, window.y
    tx, ty = x * 3 // 2, y * 3 // 2
    slices = (slice(y - 1, y + 3), slice(x - 2, x + 6))
    target_slices = (slice(ty, ty + 3), slice(tx, tx + 6))
    rx0, ry0, rx1, ry1 = (float(value) for value in asset.render_bbox)
    bbox = np.asarray(
        [
            rx0 + (x / source.width) * (rx1 - rx0),
            ry0 + (y / source.rows) * (ry1 - ry0),
            rx0 + ((x + 4) / source.width) * (rx1 - rx0),
            ry0 + ((y + 2) / source.rows) * (ry1 - ry0),
        ],
        dtype=np.float32,
    )
    return Patch(
        asset_id=asset.asset_id,
        context_glyphs=_grid(source, "glyph_ids")[slices].copy(),
        context_foreground=_grid(source, "foreground_rgb")[slices].copy(),
        context_background=_grid(source, "background_rgb")[slices].copy(),
        context_background_present=_grid(source, "background_present")[slices].copy(),
        target_glyphs=_grid(target, "glyph_ids")[target_slices].copy(),
        target_foreground=_grid(target, "foreground_rgb")[target_slices].copy(),
        target_background=_grid(target, "background_rgb")[target_slices].copy(),
        target_background_present=_grid(target, "background_present")[target_slices].copy(),
        bbox=bbox,
        scale=np.asarray([source.width / 120, source.rows / 120, target.width / 120, target.rows / 120], dtype=np.float32),
        source_lod_weights=source.lod_weights.astype(np.float32),
        target_lod_weights=target.lod_weights.astype(np.float32),
    )


class PatchSampler:
    def __init__(self, assets: list[AssetPatches], seed: int) -> None:
        if not assets:
            raise ValueError("No assets contain aligned 8x4 to 6x3 refinement windows")
        self.assets = assets
        self.random = np.random.default_rng(seed)

    def sample(self) -> Patch:
        asset = self.assets[int(self.random.integers(len(self.assets)))]
        kind = str(self.random.choice(["edge", "content", "empty"], p=[0.5, 0.4, 0.1]))
        candidates = asset.windows[kind]
        if not candidates:
            candidates = [window for values in asset.windows.values() for window in values]
        return extract_patch(asset, candidates[int(self.random.integers(len(candidates)))])

    def batches(self, batch_size: int) -> Iterator[list[Patch]]:
        while True:
            yield [self.sample() for _ in range(batch_size)]


def fixed_patches(assets: list[AssetPatches], count_per_asset: int, seed: int) -> list[Patch]:
    patches = []
    for index, asset in enumerate(assets):
        sampler = PatchSampler([asset], seed + index)
        patches.extend(sampler.sample() for _ in range(count_per_asset))
    return patches
