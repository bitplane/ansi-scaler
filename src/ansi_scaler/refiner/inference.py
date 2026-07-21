from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from ansi_scaler.ansi import AnsiCells
from ansi_scaler.dataset.reader import CompiledDataset
from ansi_scaler.refiner.config import RefinerConfig
from ansi_scaler.refiner.model import LocalAnsiRefiner


def _device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


def _lod_weights(width: int, boundaries: tuple[int, int, int], blend_radius: int = 4) -> np.ndarray:
    weights = np.zeros(4, dtype=np.float32)
    for index, boundary in enumerate(boundaries):
        start, end = boundary - blend_radius, boundary + blend_radius
        if start < width < end:
            higher = (width - start) / (2 * blend_radius)
            weights[index] = 1 - higher
            weights[index + 1] = higher
            return weights
    weights[0 if width < boundaries[0] else 1 if width < boundaries[1] else 2 if width < boundaries[2] else 3] = 1
    return weights


def load_refiner(
    checkpoint: Path, dataset: CompiledDataset, config: RefinerConfig, device: torch.device
) -> LocalAnsiRefiner:
    model = LocalAnsiRefiner(len(dataset.vocabulary["codepoints"]) + 3, config).to(device)
    if checkpoint.suffix == ".safetensors":
        state = load_file(checkpoint, device=str(device))
    else:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload.get("dataset_id") != dataset.metadata["dataset_id"]:
            raise ValueError("Checkpoint was trained against a different compiled dataset")
        if payload.get("vocabulary_sha256") != dataset.metadata["vocabulary_sha256"]:
            raise ValueError("Checkpoint was trained with a different glyph vocabulary")
        state = payload["model"]
    try:
        model.load_state_dict(state)
    except RuntimeError as error:
        raise ValueError("Checkpoint architecture does not match this visual-only refiner config") from error
    model.eval()
    return model


def scale_cells(
    cells: AnsiCells,
    *,
    width: int,
    rows: int,
    model: LocalAnsiRefiner,
    dataset: CompiledDataset,
    config: RefinerConfig,
    lod_boundaries: tuple[int, int, int],
    device: torch.device,
) -> tuple[AnsiCells, int, int]:
    """Scale an ANSI grid 1.5x by tiling learned 4x2 -> 6x3 predictions."""
    if len(cells.glyphs) != width * rows:
        raise ValueError("ANSI cells do not match the supplied dimensions")
    codepoints = dataset.vocabulary["codepoints"]
    codepoint_to_id = {value: index + 3 for index, value in enumerate(codepoints)}
    space_id = codepoint_to_id.get(ord(" "))
    if space_id is None:
        raise ValueError("Training vocabulary does not contain a space glyph")

    padded_width = math.ceil(width / 4) * 4
    padded_rows = math.ceil(rows / 2) * 2
    grid_shape = (padded_rows + 2, padded_width + 4)
    glyphs = np.full(grid_shape, space_id, dtype=np.int64)
    foreground = np.zeros((*grid_shape, 3), dtype=np.float32)
    background = np.zeros((*grid_shape, 3), dtype=np.float32)
    present = np.zeros(grid_shape, dtype=np.float32)
    source_slice = (slice(1, rows + 1), slice(2, width + 2))
    source_glyphs = np.asarray([codepoint_to_id.get(int(value), 2) for value in cells.glyphs]).reshape(rows, width)
    glyphs[source_slice] = source_glyphs
    foreground[source_slice] = cells.foreground_rgb.reshape(rows, width, 3) / 255
    background[source_slice] = cells.background_rgb.reshape(rows, width, 3) / 255
    present[source_slice] = cells.background_present.reshape(rows, width)

    contexts, metadata = [], []
    source_lod = _lod_weights(width, lod_boundaries)
    target_width = math.ceil(width * 1.5)
    target_rows = math.ceil(rows * 1.5)
    target_lod = _lod_weights(target_width, lod_boundaries)
    for y in range(0, padded_rows, 2):
        for x in range(0, padded_width, 4):
            contexts.append((slice(y, y + 4), slice(x, x + 8)))
            metadata.append(
                [
                    x / width,
                    y / rows,
                    (x + 4) / width,
                    (y + 2) / rows,
                    width / 120,
                    rows / 120,
                    target_width / 120,
                    target_rows / 120,
                    *source_lod,
                    *target_lod,
                ]
            )
    batch_glyphs = torch.from_numpy(np.stack([glyphs[item] for item in contexts]).reshape(-1, 32)).to(device)
    batch_foreground = torch.from_numpy(np.stack([foreground[item] for item in contexts]).reshape(-1, 32, 3)).to(device)
    batch_background = torch.from_numpy(np.stack([background[item] for item in contexts]).reshape(-1, 32, 3)).to(device)
    batch_present = torch.from_numpy(np.stack([present[item] for item in contexts]).reshape(-1, 32)).to(device)
    batch_metadata = torch.tensor(np.asarray(metadata), dtype=torch.float32, device=device)
    with torch.inference_mode():
        output = model(batch_glyphs, batch_foreground, batch_background, batch_present, batch_metadata)

    output_width, output_rows = padded_width * 3 // 2, padded_rows * 3 // 2
    output_glyphs = output.glyph_logits.argmax(-1).cpu().numpy().reshape(-1, 3, 6)
    output_foreground = (output.foreground.cpu().numpy().reshape(-1, 3, 6, 3) * 255).round().astype(np.uint8)
    output_background = (output.background.cpu().numpy().reshape(-1, 3, 6, 3) * 255).round().astype(np.uint8)
    output_present = output.background_logits.sigmoid().cpu().numpy().reshape(-1, 3, 6) >= 0.5
    assembled_glyphs = np.empty((output_rows, output_width), dtype=np.uint32)
    assembled_foreground = np.empty((output_rows, output_width, 3), dtype=np.uint8)
    assembled_background = np.empty((output_rows, output_width, 3), dtype=np.uint8)
    assembled_present = np.empty((output_rows, output_width), dtype=np.bool_)
    index = 0
    for y in range(0, output_rows, 3):
        for x in range(0, output_width, 6):
            ids = output_glyphs[index]
            assembled_glyphs[y : y + 3, x : x + 6] = np.asarray(
                [codepoints[value - 3] if 3 <= value < len(codepoints) + 3 else ord("?") for value in ids.flat]
            ).reshape(3, 6)
            assembled_foreground[y : y + 3, x : x + 6] = output_foreground[index]
            assembled_background[y : y + 3, x : x + 6] = output_background[index]
            assembled_present[y : y + 3, x : x + 6] = output_present[index]
            index += 1
    cropped = (slice(0, target_rows), slice(0, target_width))
    return (
        AnsiCells(
            glyphs=assembled_glyphs[cropped].reshape(-1),
            foreground_rgb=assembled_foreground[cropped].reshape(-1, 3),
            background_rgb=assembled_background[cropped].reshape(-1, 3),
            background_present=assembled_present[cropped].reshape(-1),
        ),
        target_width,
        target_rows,
    )
