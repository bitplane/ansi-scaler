from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np
import zstandard

from ansi_scaler.ansi import AnsiCells, decode_ansi, encode_ansi
from ansi_scaler.dataset.compiler import compile_dataset
from ansi_scaler.dataset.models import DatasetRecipe
from ansi_scaler.dataset.reader import CompiledDataset
from ansi_scaler.dataset.selection import assign_split
from ansi_scaler.dataset.validate import validate_dataset
from ansi_scaler.manifests import write_jsonl


def test_ansi_cells_round_trip_supplementary_glyph_and_transparency() -> None:
    cells = AnsiCells(
        glyphs=np.asarray([0x1FB8C, ord(" ")], dtype=np.uint32),
        foreground_rgb=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
        background_rgb=np.asarray([[7, 8, 9], [0, 0, 0]], dtype=np.uint8),
        background_present=np.asarray([True, False], dtype=np.bool_),
    )
    encoded = encode_ansi(cells, width=2, rows=1)
    decoded = decode_ansi(encoded, width=2, rows=1)
    np.testing.assert_array_equal(decoded.glyphs, cells.glyphs)
    np.testing.assert_array_equal(decoded.foreground_rgb, cells.foreground_rgb)
    np.testing.assert_array_equal(decoded.background_rgb, cells.background_rgb)
    np.testing.assert_array_equal(decoded.background_present, cells.background_present)


def test_prompt_family_split_is_stable() -> None:
    splits = type("Splits", (), {"train": 0.8, "validation": 0.1})()
    assert assign_split("family", 42, splits) == assign_split("family", 42, splits)


def _archive(path: Path, ansi: bytes) -> tuple[str, dict[str, object]]:
    level = {
        "width": 2,
        "rows": 1,
        "source_lods": [{"name": "lod-3", "weight": 1.0}],
        "path": "levels/002.ansi",
        "sha256": hashlib.sha256(ansi).hexdigest(),
        "bytes": len(ansi),
    }
    path.parent.mkdir(parents=True)
    with path.open("wb") as raw:
        with zstandard.ZstdCompressor().stream_writer(raw, closefd=False) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                info = tarfile.TarInfo(level["path"])
                info.size = len(ansi)
                archive.addfile(info, io.BytesIO(ansi))
    return hashlib.sha256(path.read_bytes()).hexdigest(), level


def test_compile_read_and_validate_tiny_dataset(tmp_path: Path) -> None:
    data = tmp_path / "data"
    manifests = data / "runs" / "tiny" / "manifests"
    artifact = data / "artifacts" / "pyramids" / "aa" / "asset.tar.zst"
    ansi = encode_ansi(
        AnsiCells(
            glyphs=np.asarray([ord("A"), 0x1FB8C], dtype=np.uint32),
            foreground_rgb=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
            background_rgb=np.asarray([[0, 0, 0], [7, 8, 9]], dtype=np.uint8),
            background_present=np.asarray([False, True], dtype=np.bool_),
        ),
        width=2,
        rows=1,
    )
    archive_sha, level = _archive(artifact, ansi)
    common = {"concept_id": "thing", "concept_name": "thing", "prompt_id": "family", "prompt": "thing"}
    prompt = {**common, "id": "p", "stage": "prompts"}
    raster = {**common, "id": "g", "parent_id": "p", "stage": "generate"}
    cutout = {**common, "id": "r", "parent_id": "g", "stage": "background"}
    lod = {**common, "id": "l", "parent_id": "r", "stage": "lod"}
    pyramid = {
        **common,
        "id": "a",
        "parent_id": "l",
        "stage": "pyramid",
        "artifact": artifact.relative_to(data).as_posix(),
        "archive_sha256": archive_sha,
        "pyramid_format": "ansi-scaler-pyramid-v3",
        "pyramid_levels": [level],
        "geometry": {
            "content_bbox": [0.1, 0.2, 0.8, 0.9],
            "render_bbox": [0.0, 0.1, 0.9, 1.0],
            "canvas_size": [512, 512],
            "render_size_px": [460, 460],
        },
    }
    classification = {**common, "id": "c", "parent_id": "r", "stage": "classify"}
    verification = {
        **common,
        "id": "v",
        "parent_id": "c",
        "stage": "verify",
        "verification": {"decision": "accept"},
    }
    for name, records in (
        ("prompts", [prompt]),
        ("rasters", [raster]),
        ("backgrounds", [cutout]),
        ("lods", [lod]),
        ("pyramids", [pyramid]),
        ("classifications", [classification]),
        ("verifications", [verification]),
    ):
        write_jsonl(manifests / f"{name}.jsonl", records)
    run_config = tmp_path / "run.yaml"
    run_config.write_text(
        "\n".join(
            [
                "name: tiny",
                f"data_dir: {data}",
                "lod:",
                "  levels: []",
            ]
        )
    )
    recipe = DatasetRecipe(
        name="tiny",
        run_config=run_config,
        output_root=tmp_path / "datasets",
        shard_count=2,
    )
    destination = compile_dataset(recipe)
    dataset = CompiledDataset.open(destination)
    loaded = dataset.asset("p").level(2)
    assert loaded.glyph_ids.tolist() == [3, 4]
    assert loaded.background_present.tolist() == [False, True]
    assert validate_dataset(destination) == {"assets": 1, "levels": 1, "cells": 2}
    assert compile_dataset(recipe) == destination
