from __future__ import annotations

import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import zstandard
from safetensors.numpy import save_file
from tqdm.auto import tqdm

from ansi_scaler.ansi import decode_ansi
from ansi_scaler.config import load_run_config
from ansi_scaler.dataset.models import DatasetRecipe
from ansi_scaler.dataset.selection import assign_split, public_selection, select_assets
from ansi_scaler.identity import sha256_file, stable_id
from ansi_scaler.manifests import write_jsonl


COMPILER_VERSION = "ansi-dataset-compiler-v1"
VOCABULARY_VERSION = "ansi-glyph-vocabulary-v1"
LOD_NAMES = ("lod-3", "lod-2", "lod-1", "lod-0")


def _archive_levels(path: Path, record: dict[str, Any]) -> Iterator[tuple[dict[str, Any], bytes]]:
    expected = {level["path"]: level for level in record["pyramid_levels"]}
    found: set[str] = set()
    with path.open("rb") as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                for member in archive:
                    level = expected.get(member.name)
                    if level is None:
                        continue
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise ValueError(f"Cannot read {member.name} from {path}")
                    data = handle.read()
                    if hashlib.sha256(data).hexdigest() != level["sha256"]:
                        raise ValueError(f"Pyramid level hash mismatch: {path}:{member.name}")
                    found.add(member.name)
                    yield level, data
    missing = expected.keys() - found
    if missing:
        raise ValueError(f"Pyramid archive is missing levels: {', '.join(sorted(missing))}")


def _load_base_vocabulary(path: Path | None) -> tuple[list[int | None], str | None]:
    if path is None:
        return [None, None, None], None
    payload = json.loads(path.read_text())
    if payload.get("format") != VOCABULARY_VERSION:
        raise ValueError(f"Unsupported base vocabulary: {path}")
    return [None, None, None, *payload["codepoints"]], sha256_file(path)


def _build_vocabulary(recipe: DatasetRecipe, config: Any, assets: list[dict[str, Any]]) -> dict[str, Any]:
    entries, parent_sha = _load_base_vocabulary(recipe.base_vocabulary)
    existing = {value for value in entries if value is not None}
    observed: set[int] = set()
    for asset in tqdm(assets, desc="vocabulary", unit="pyramid", dynamic_ncols=True):
        archive = config.data_dir / asset["pyramid_artifact"]
        expected_archive_hash = asset.get("pyramid_sha256")
        if expected_archive_hash and sha256_file(archive) != expected_archive_hash:
            raise ValueError(f"Pyramid archive hash mismatch: {archive}")
        for level, data in _archive_levels(archive, asset["pyramid"]):
            observed.update(int(value) for value in decode_ansi(data, width=level["width"], rows=level["rows"]).glyphs)
    entries.extend(sorted(observed - existing))
    if len(entries) > np.iinfo(np.uint16).max + 1:
        raise ValueError("Glyph vocabulary exceeds uint16 capacity")
    return {
        "format": VOCABULARY_VERSION,
        "reserved": {"PAD": 0, "MASK": 1, "UNK": 2},
        "codepoints": entries[3:],
        "base_vocabulary_sha256": parent_sha,
    }


def _weights(level: dict[str, Any]) -> list[float]:
    values = {item["name"]: float(item["weight"]) for item in level.get("source_lods", [])}
    result = [values.get(name, 0.0) for name in LOD_NAMES]
    if not np.isclose(sum(result), 1.0):
        raise ValueError(f"LOD weights do not sum to one at width {level['width']}")
    return result


def _compile_shard(
    path: Path, assets: list[dict[str, Any]], config: Any, vocabulary: dict[str, Any], shard_index: int
) -> list[dict[str, Any]]:
    codepoint_to_id = {value: index + 3 for index, value in enumerate(vocabulary["codepoints"])}
    glyphs: list[np.ndarray] = []
    foreground: list[np.ndarray] = []
    background: list[np.ndarray] = []
    background_present: list[np.ndarray] = []
    level_offsets = [0]
    widths: list[int] = []
    rows: list[int] = []
    lod_weights: list[list[float]] = []
    asset_offsets = [0]
    content_bbox: list[list[float]] = []
    render_bbox: list[list[float]] = []
    canvas_size: list[list[int]] = []
    render_size: list[list[int]] = []
    index_records: list[dict[str, Any]] = []
    for local_index, asset in enumerate(assets):
        archive = config.data_dir / asset["pyramid_artifact"]
        level_count = 0
        for level, data in sorted(_archive_levels(archive, asset["pyramid"]), key=lambda item: item[0]["width"]):
            cells = decode_ansi(data, width=level["width"], rows=level["rows"])
            try:
                ids = np.asarray([codepoint_to_id[int(value)] for value in cells.glyphs], dtype=np.uint16)
            except KeyError as error:
                raise ValueError(f"Glyph U+{int(error.args[0]):04X} is absent from the frozen vocabulary") from error
            glyphs.append(ids)
            foreground.append(cells.foreground_rgb)
            background.append(cells.background_rgb)
            background_present.append(cells.background_present)
            level_offsets.append(level_offsets[-1] + len(ids))
            widths.append(level["width"])
            rows.append(level["rows"])
            lod_weights.append(_weights(level))
            level_count += 1
        geometry = asset["pyramid"]["geometry"]
        content_bbox.append(geometry["content_bbox"])
        render_bbox.append(geometry["render_bbox"])
        canvas_size.append(geometry["canvas_size"])
        render_size.append(geometry["render_size_px"])
        asset_offsets.append(asset_offsets[-1] + level_count)
        index_records.append(
            {
                **public_selection(asset),
                "shard": shard_index,
                "local_index": local_index,
                "level_start": asset_offsets[-2],
                "level_count": level_count,
            }
        )
    tensors = {
        "glyph_ids": np.concatenate(glyphs) if glyphs else np.empty(0, dtype=np.uint16),
        "foreground_rgb": np.concatenate(foreground) if foreground else np.empty((0, 3), dtype=np.uint8),
        "background_rgb": np.concatenate(background) if background else np.empty((0, 3), dtype=np.uint8),
        "background_present": np.concatenate(background_present) if background_present else np.empty(0, dtype=np.bool_),
        "level_cell_offsets": np.asarray(level_offsets, dtype=np.int64),
        "level_widths": np.asarray(widths, dtype=np.uint16),
        "level_rows": np.asarray(rows, dtype=np.uint16),
        "level_lod_weights": np.asarray(lod_weights, dtype=np.float32).reshape((-1, 4)),
        "asset_level_offsets": np.asarray(asset_offsets, dtype=np.int64),
        "content_bbox": np.asarray(content_bbox, dtype=np.float32).reshape((-1, 4)),
        "render_bbox": np.asarray(render_bbox, dtype=np.float32).reshape((-1, 4)),
        "canvas_size": np.asarray(canvas_size, dtype=np.uint32).reshape((-1, 2)),
        "render_size": np.asarray(render_size, dtype=np.uint32).reshape((-1, 2)),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, temporary, metadata={"format": "ansi-pyramid-tensors-v1", "lod_order": ",".join(LOD_NAMES)})
    temporary.replace(path)
    return index_records


def prepare(recipe: DatasetRecipe, *, limit: int | None = None) -> tuple[Any, list[dict[str, Any]], str]:
    config = load_run_config(recipe.run_config)
    selection = select_assets(config, recipe.pyramid_format, limit=limit)
    for record in selection:
        record["split"] = assign_split(record["prompt_family_id"], recipe.split_seed, recipe.splits)
    identity_selection = [public_selection(record) for record in selection]
    base_vocabulary_sha256 = sha256_file(recipe.base_vocabulary) if recipe.base_vocabulary else None
    dataset_id = stable_id(
        COMPILER_VERSION,
        recipe.model_dump(mode="json"),
        base_vocabulary_sha256,
        identity_selection,
        limit,
    )
    return config, selection, dataset_id


def plan_dataset(recipe: DatasetRecipe, *, limit: int | None = None) -> dict[str, Any]:
    _config, selection, dataset_id = prepare(recipe, limit=limit)
    reasons = Counter(record["selection_reason"] for record in selection)
    splits = Counter(record["split"] for record in selection if record["include"])
    estimated_cells = sum(
        level["width"] * level["rows"]
        for record in selection
        if record["include"]
        for level in record["pyramid"]["pyramid_levels"]
    )
    return {
        "dataset_id": dataset_id,
        "total": len(selection),
        "included": sum(record["include"] for record in selection),
        "excluded": sum(not record["include"] for record in selection),
        "selection_reasons": dict(sorted(reasons.items())),
        "splits": dict(sorted(splits.items())),
        "estimated_cells": estimated_cells,
        "estimated_tensor_bytes": estimated_cells * 9,
    }


def compile_dataset(recipe: DatasetRecipe, *, limit: int | None = None) -> Path:
    config, selection, dataset_id = prepare(recipe, limit=limit)
    destination = recipe.output_root / recipe.name / dataset_id
    building = destination.with_name(destination.name + ".building")
    if destination.exists():
        return destination
    building.mkdir(parents=True, exist_ok=True)
    write_jsonl(building / "selection.jsonl", (public_selection(record) for record in selection))
    included = [record for record in selection if record["include"]]
    vocabulary_path = building / "vocabulary.json"
    if vocabulary_path.exists():
        vocabulary = json.loads(vocabulary_path.read_text())
    else:
        vocabulary = _build_vocabulary(recipe, config, included)
        vocabulary_path.write_text(json.dumps(vocabulary, sort_keys=True, indent=2) + "\n")
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(recipe.shard_count)]
    for record in included:
        bucket = int(stable_id("dataset-shard-v1", record["asset_id"])[:16], 16) % recipe.shard_count
        buckets[bucket].append(record)
    all_indexes: list[dict[str, Any]] = []
    for shard_index, assets in enumerate(tqdm(buckets, desc="compile", unit="shard", dynamic_ncols=True)):
        shard = building / f"shard-{shard_index:05d}.safetensors"
        index_path = building / f"shard-{shard_index:05d}.jsonl"
        if not shard.exists() or not index_path.exists():
            records = _compile_shard(shard, assets, config, vocabulary, shard_index)
            write_jsonl(index_path, records)
        all_indexes.extend(json.loads(line) for line in index_path.read_text().splitlines() if line.strip())
    write_jsonl(building / "index.jsonl", sorted(all_indexes, key=lambda record: record["asset_id"]))
    metadata = {
        "format": recipe.format,
        "compiler": COMPILER_VERSION,
        "dataset_id": dataset_id,
        "name": recipe.name,
        "recipe": recipe.model_dump(mode="json"),
        "limit": limit,
        "asset_count": len(included),
        "shard_count": recipe.shard_count,
        "vocabulary_sha256": sha256_file(vocabulary_path),
        "selection_sha256": sha256_file(building / "selection.jsonl"),
        "index_sha256": sha256_file(building / "index.jsonl"),
        "shards": [
            {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(building.glob("shard-*.safetensors"))
        ],
    }
    (building / "dataset.json").write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    destination.parent.mkdir(parents=True, exist_ok=True)
    building.replace(destination)
    return destination
