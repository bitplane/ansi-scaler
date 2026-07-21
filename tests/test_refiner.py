from __future__ import annotations

import numpy as np
import torch

from ansi_scaler.dataset.reader import CompiledLevel
from ansi_scaler.refiner.config import RefinerConfig
from ansi_scaler.refiner.model import LocalAnsiRefiner, refiner_loss
from ansi_scaler.refiner.sampler import AssetPatches, aligned_windows, exact_pairs, extract_patch


def level(width: int, rows: int) -> CompiledLevel:
    glyphs = np.full(width * rows, 3, dtype=np.uint16)
    glyphs.reshape(rows, width)[rows // 3 : rows * 2 // 3, width // 3 : width * 2 // 3] = 4
    return CompiledLevel(
        width=width,
        rows=rows,
        glyph_ids=glyphs,
        foreground_rgb=np.full((width * rows, 3), 100, dtype=np.uint8),
        background_rgb=np.full((width * rows, 3), 20, dtype=np.uint8),
        background_present=(glyphs != 3),
        lod_weights=np.asarray([0, 0, 1, 0], dtype=np.float32),
    )


def test_exact_aligned_patch_has_context_target_and_original_bbox() -> None:
    levels = {10: level(10, 6), 15: level(15, 9), 16: level(16, 9)}
    assert [(source.width, target.width) for source, target in exact_pairs(levels)] == [(10, 15)]
    windows = aligned_windows(levels, space_id=3)
    window = next(value for values in windows.values() for value in values if value.x == 2 and value.y == 2)
    asset = AssetPatches(
        asset_id="asset", split="train",
        render_bbox=np.asarray([0.1, 0.2, 0.9, 0.8], dtype=np.float32),
        levels=levels, windows=windows,
    )

    patch = extract_patch(asset, window)

    assert patch.context_glyphs.shape == (4, 8)
    assert patch.target_glyphs.shape == (3, 6)
    np.testing.assert_allclose(patch.bbox, [0.26, 0.4, 0.58, 0.6], atol=1e-6)


def test_refiner_forward_and_masked_loss_backpropagate() -> None:
    config = RefinerConfig(
        name="test", dataset_recipe="dataset.yaml",
        d_model=32, heads=4, context_layers=1, decoder_layers=1,
    )
    model = LocalAnsiRefiner(vocabulary_size=8, config=config)
    batch = 2
    output = model(
        torch.randint(3, 8, (batch, 32)),
        torch.rand(batch, 32, 3),
        torch.rand(batch, 32, 3),
        torch.randint(0, 2, (batch, 32)).float(),
        torch.rand(batch, 16),
    )
    target_glyphs = torch.randint(3, 8, (batch, 18))
    loss, parts = refiner_loss(
        output, target_glyphs, torch.rand(batch, 18, 3), torch.rand(batch, 18, 3),
        torch.randint(0, 2, (batch, 18)).float(), 3, config,
    )
    loss.backward()

    assert output.glyph_logits.shape == (batch, 18, 8)
    assert output.foreground.shape == (batch, 18, 3)
    assert set(parts) == {"loss", "glyph", "foreground", "background", "presence"}
    assert torch.isfinite(loss)
