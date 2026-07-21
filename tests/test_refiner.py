from __future__ import annotations

import numpy as np
import torch
from mlflow import MlflowClient
from types import SimpleNamespace

from ansi_scaler.ansi import AnsiCells
from ansi_scaler.dataset.reader import CompiledLevel
from ansi_scaler.refiner.config import RefinerConfig
from ansi_scaler.refiner.inference import _lod_weights, scale_cells
from ansi_scaler.refiner.model import LocalAnsiRefiner, refiner_loss
from ansi_scaler.refiner.sampler import AssetPatches, aligned_windows, exact_pairs, extract_patch
from ansi_scaler.refiner.tracking import RefinerTracker, default_tracking_uri


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


def test_whole_grid_inference_pads_and_crops_arbitrary_dimensions() -> None:
    config = RefinerConfig(
        name="test", dataset_recipe="dataset.yaml",
        d_model=32, heads=4, context_layers=1, decoder_layers=1,
    )
    vocabulary = {"codepoints": [ord(" "), ord("#")]}
    dataset = SimpleNamespace(vocabulary=vocabulary)
    model = LocalAnsiRefiner(vocabulary_size=5, config=config)
    width, rows = 5, 3
    cells = AnsiCells(
        glyphs=np.full(width * rows, ord("#"), dtype=np.uint32),
        foreground_rgb=np.full((width * rows, 3), 200, dtype=np.uint8),
        background_rgb=np.zeros((width * rows, 3), dtype=np.uint8),
        background_present=np.zeros(width * rows, dtype=np.bool_),
    )

    enlarged, enlarged_width, enlarged_rows = scale_cells(
        cells, width=width, rows=rows, model=model, dataset=dataset, config=config,
        lod_boundaries=(10, 40, 80), device=torch.device("cpu"),
    )

    assert (enlarged_width, enlarged_rows) == (8, 5)
    assert enlarged.glyphs.shape == (40,)
    assert enlarged.foreground_rgb.shape == (40, 3)


def test_lod_weights_blend_across_boundaries() -> None:
    np.testing.assert_allclose(_lod_weights(10, (10, 40, 80)), [0.5, 0.5, 0, 0])
    np.testing.assert_allclose(_lod_weights(20, (10, 40, 80)), [0, 1, 0, 0])


def test_mlflow_tracker_records_metrics_parameters_and_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "report.json"
    artifact.write_text("{}\n")
    tracker = RefinerTracker.start(
        output_root=tmp_path,
        experiment="test-refiner",
        run_dir=run_dir,
        run_name="test-run",
        log_steps=10,
        parameters={"config": {"steps": 20}, "dataset": {"id": "dataset-id"}},
        tags={"ansi_scaler.run_id": "local-id"},
    )
    tracker.metrics({"loss": 2.0}, step=1, prefix="train")
    tracker.metrics({"loss": 1.0}, step=10, prefix="train")
    tracker.artifact(artifact)
    tracker.finish()

    client = MlflowClient(tracking_uri=default_tracking_uri(tmp_path))
    run = client.get_run(tracker.run_id)
    history = client.get_metric_history(tracker.run_id, "train/loss")
    assert run.info.status == "FINISHED"
    assert run.data.params["config.steps"] == "20"
    assert [(metric.step, metric.value) for metric in history] == [(10, 1.0)]
    assert [item.path for item in client.list_artifacts(tracker.run_id)] == ["report.json"]
