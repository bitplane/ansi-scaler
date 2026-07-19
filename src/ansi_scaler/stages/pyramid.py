from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import tarfile
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any

import cairosvg
import chuda
import zstandard
from PIL import Image
from tqdm.auto import tqdm

from ansi_scaler.artifacts import artifact_path, atomic_destination
from ansi_scaler.config import RunConfig
from ansi_scaler.identity import sha256_file, stable_id
from ansi_scaler.manifests import append_jsonl, known_ids, read_jsonl, relative_path, resolve_path
from ansi_scaler.runner import StageInfrastructureError


PYRAMID_FORMAT = "ansi-scaler-pyramid-v2"
INPUT_RASTERIZATION_CONTRACT = "shared-crop-rgba-v1"


@lru_cache(maxsize=1)
def rasterizer_signature() -> dict[str, str]:
    return {
        "contract": INPUT_RASTERIZATION_CONTRACT,
        "cairosvg_version": importlib.metadata.version("cairosvg"),
        "pillow_version": importlib.metadata.version("pillow"),
        "mode": "RGBA",
    }


def source_for_width(record: dict[str, Any], width: int, config: RunConfig) -> tuple[str, Path]:
    settings = config.chuda
    if width < settings.lod_3_below:
        level_name = "lod-3"
    elif width < settings.lod_2_below:
        level_name = "lod-2"
    elif width < settings.lod_1_below:
        level_name = "lod-1"
    else:
        return "original", resolve_path(record["original"], config.data_dir)
    level = next(level for level in record["levels"] if level["name"] == level_name)
    return level_name, resolve_path(level["svg"], config.data_dir)


def pyramid_id(record: dict[str, Any], config: RunConfig) -> str:
    settings = config.chuda.model_dump(mode="json", exclude={"backend"})
    return stable_id(PYRAMID_FORMAT, record["id"], settings, rasterizer_signature())


def object_geometry(record: dict[str, Any], config: RunConfig) -> dict[str, Any]:
    original = resolve_path(record["original"], config.data_dir)
    with Image.open(original) as image:
        canvas_width, canvas_height = image.size
        alpha = image.getchannel("A") if image.mode == "RGBA" else Image.new("L", image.size, 255)
        threshold = config.chuda.alpha_bbox_threshold
        mask = alpha.point(lambda value: 255 if value > threshold else 0)
        content_bbox = mask.getbbox()
    if content_bbox is None:
        raise ValueError(f"Cutout has no alpha above threshold {threshold}: {original}")

    left, top, right, bottom = content_bbox
    padding_x = math.ceil((right - left) * config.chuda.crop_padding_fraction)
    padding_y = math.ceil((bottom - top) * config.chuda.crop_padding_fraction)
    render_bbox = (
        max(0, left - padding_x),
        max(0, top - padding_y),
        min(canvas_width, right + padding_x),
        min(canvas_height, bottom + padding_y),
    )

    def normalise(bbox: tuple[int, int, int, int]) -> list[float]:
        return [
            bbox[0] / canvas_width,
            bbox[1] / canvas_height,
            bbox[2] / canvas_width,
            bbox[3] / canvas_height,
        ]

    return {
        "canvas_size": [canvas_width, canvas_height],
        "content_bbox_px": list(content_bbox),
        "content_bbox": normalise(content_bbox),
        "render_bbox_px": list(render_bbox),
        "render_size_px": [render_bbox[2] - render_bbox[0], render_bbox[3] - render_bbox[1]],
        "render_bbox": normalise(render_bbox),
        "alpha_bbox_threshold": threshold,
        "crop_padding_fraction": config.chuda.crop_padding_fraction,
    }


def _check_chuda(config: RunConfig) -> None:
    actual = importlib.metadata.version("chuda-ansi")
    if actual != config.chuda.version:
        raise RuntimeError(f"Expected Chuda {config.chuda.version}, found {actual}")


def _normalise_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    return info


def _write_archive(destination: Path, metadata: dict[str, Any], files: list[tuple[bytes, str]]) -> None:
    with atomic_destination(destination) as temporary:
        compressor = zstandard.ZstdCompressor(threads=-1)
        with temporary.open("wb") as raw:
            with compressor.stream_writer(raw, closefd=False) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as archive:
                    encoded = json.dumps(metadata, sort_keys=True, indent=2).encode()
                    info = _normalise_tar_info(tarfile.TarInfo("metadata.json"))
                    info.size = len(encoded)
                    archive.addfile(info, io.BytesIO(encoded))
                    for data, arcname in files:
                        info = _normalise_tar_info(tarfile.TarInfo(arcname))
                        info.size = len(data)
                        archive.addfile(info, io.BytesIO(data))


def _crop_record_sources(
    record: dict[str, Any], config: RunConfig, geometry: dict[str, Any], source_names: tuple[str, ...]
) -> dict[str, chuda.Image]:
    canvas_size = geometry["canvas_size"]
    render_bbox = tuple(geometry["render_bbox_px"])
    result: dict[str, chuda.Image] = {}
    for source_name in source_names:
        if source_name == "original":
            source = resolve_path(record["original"], config.data_dir)
            with Image.open(source) as opened:
                image = opened.convert("RGBA")
        else:
            level = next(level for level in record["levels"] if level["name"] == source_name)
            source = resolve_path(level["svg"], config.data_dir)
            encoded = cairosvg.svg2png(url=str(source), output_width=canvas_size[0], output_height=canvas_size[1])
            with Image.open(io.BytesIO(encoded)) as opened:
                image = opened.convert("RGBA")
        if image.size != tuple(canvas_size):
            raise ValueError(f"Prepared source canvas mismatch: expected {tuple(canvas_size)}, got {image.size}")
        cropped = image.crop(render_bbox)
        result[source_name] = chuda.Image.from_rgba(cropped.width, cropped.height, cropped.tobytes())
    return result


def run_pyramid(
    config: RunConfig,
    *,
    limit: int | None = None,
    force: bool = False,
    retry_errors: bool = False,
) -> tuple[int, int, int]:
    if config.chuda.min_width > config.chuda.max_width:
        raise ValueError("chuda.min_width must not exceed chuda.max_width")
    if not config.chuda.lod_3_below < config.chuda.lod_2_below < config.chuda.lod_1_below:
        raise ValueError("Chuda LOD thresholds must be strictly increasing")
    output_manifest = config.manifest_dir / "pyramids.jsonl"
    error_manifest = config.manifest_dir / "pyramids.errors.jsonl"
    if force:
        output_manifest.unlink(missing_ok=True)
        error_manifest.unlink(missing_ok=True)
    completed = known_ids(output_manifest)
    failed_parents = {record["parent_id"] for record in read_jsonl(error_manifest)} if not retry_errors else set()
    unique_records = {record["id"]: record for record in read_jsonl(config.manifest_dir / "lods.jsonl")}
    selected = list(unique_records.values())[: limit or config.limit]
    pending = [
        record
        for record in selected
        if pyramid_id(record, config) not in completed and record["id"] not in failed_parents
    ]
    skipped = len(selected) - len(pending)
    if not pending:
        return 0, 0, skipped

    _check_chuda(config)
    widths = tuple(range(config.chuda.min_width, config.chuda.max_width + 1))
    source_names = tuple(dict.fromkeys(source_for_width(pending[0], width, config)[0] for width in widths))
    renderer = chuda.Renderer(config.chuda.backend, config.chuda.max_batch_cells)
    successes = 0
    failures = 0
    rendered_cells = 0
    progress = tqdm(total=len(pending), desc="pyramids", unit="pyramid", dynamic_ncols=True)
    progress.set_postfix(completed=0, failed=0, skipped=skipped, cells=0, refresh=False)
    try:
        for record in pending:
            output_id = pyramid_id(record, config)
            try:
                geometry = object_geometry(record, config)
                sources = _crop_record_sources(record, config, geometry, source_names)
                requests = [(sources[source_for_width(record, width, config)[0]], width) for width in widths]
                frames = renderer.render_many(
                    requests,
                    font_ratio=config.chuda.font_ratio,
                    transparent_threshold=config.chuda.transparent_threshold,
                )
                levels = []
                files = []
                backends = set()
                for width, frame in zip(widths, frames, strict=True):
                    data = bytes(frame.to_ansi())
                    source_name, _ = source_for_width(record, width, config)
                    path = f"levels/{width:03d}.ansi"
                    levels.append(
                        {
                            "width": width,
                            "rows": frame.rows,
                            "source_lod": source_name,
                            "bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "path": path,
                        }
                    )
                    files.append((data, path))
                    backends.add(frame.backend)
                    rendered_cells += frame.columns * frame.rows
                metadata = {
                    "format": PYRAMID_FORMAT,
                    "id": output_id,
                    "parent_id": record["id"],
                    "chuda": config.chuda.model_dump(mode="json"),
                    "chuda_backends": sorted(backends),
                    "input_rasterization": rasterizer_signature(),
                    "geometry": geometry,
                    "levels": levels,
                }
                destination = artifact_path(config.artifact_dir, "pyramids", output_id, ".tar.zst")
                _write_archive(destination, metadata, files)
                append_jsonl(
                    output_manifest,
                    {
                        **record,
                        "id": output_id,
                        "parent_id": record["id"],
                        "stage": "pyramid",
                        "artifact": relative_path(destination, config.data_dir),
                        "archive_sha256": sha256_file(destination),
                        "chuda_version": config.chuda.version,
                        "chuda_backends": metadata["chuda_backends"],
                        "pyramid_format": metadata["format"],
                        "input_rasterization": metadata["input_rasterization"],
                        "geometry": geometry,
                        "pyramid_levels": levels,
                    },
                )
                successes += 1
            except (MemoryError, OSError) as error:
                raise StageInfrastructureError(f"Pyramid infrastructure failed: {error}") from error
            except RuntimeError as error:
                if config.chuda.backend == "cuda":
                    raise StageInfrastructureError(f"Chuda CUDA backend failed: {error}") from error
                append_jsonl(
                    error_manifest,
                    {
                        "parent_id": record["id"],
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
                failures += 1
            except Exception as error:  # noqa: BLE001 - reject only the malformed source record
                append_jsonl(
                    error_manifest,
                    {
                        "parent_id": record["id"],
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
                failures += 1
            progress.update()
            progress.set_postfix(
                completed=successes, failed=failures, skipped=skipped, cells=rendered_cells, refresh=False
            )
    finally:
        progress.close()
    return successes, failures, skipped
