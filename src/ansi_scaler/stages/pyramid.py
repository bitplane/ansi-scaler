from __future__ import annotations

import io
import json
import math
import shutil
import subprocess
import tarfile
import tempfile
import traceback
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm.auto import tqdm

from ansi_scaler.artifacts import artifact_path, atomic_destination
from ansi_scaler.config import RunConfig
from ansi_scaler.identity import sha256_file, stable_id
from ansi_scaler.manifests import append_jsonl, known_ids, read_jsonl, relative_path, resolve_path


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
    return level_name, resolve_path(level["preview"], config.data_dir)


def pyramid_id(record: dict[str, Any], config: RunConfig) -> str:
    return stable_id("chuda-pyramid-v1", record["id"], config.chuda.model_dump(mode="json"))


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
        "render_bbox": normalise(render_bbox),
        "alpha_bbox_threshold": threshold,
        "crop_padding_fraction": config.chuda.crop_padding_fraction,
    }


def _write_cropped_source(source: Path, destination: Path, render_bbox: list[float]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        width, height = image.size
        crop_bbox = (
            max(0, math.floor(render_bbox[0] * width)),
            max(0, math.floor(render_bbox[1] * height)),
            min(width, math.ceil(render_bbox[2] * width)),
            min(height, math.ceil(render_bbox[3] * height)),
        )
        image.crop(crop_bbox).save(destination, format="PNG")


def _check_chuda(config: RunConfig) -> None:
    executable = shutil.which(config.chuda.executable)
    if executable is None:
        raise FileNotFoundError(f"Chuda executable was not found on PATH: {config.chuda.executable}")
    result = subprocess.run([executable, "--version"], check=True, capture_output=True, text=True)
    actual = result.stdout.strip().removeprefix("chuda ")
    if actual != config.chuda.version:
        raise RuntimeError(f"Expected Chuda {config.chuda.version}, found {actual}")
    if shutil.which("zstd") is None:
        raise FileNotFoundError("zstd was not found on PATH")


def _normalise_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    return info


def _write_archive(destination: Path, metadata: dict[str, Any], files: list[tuple[Path, str]]) -> None:
    with atomic_destination(destination) as temporary:
        compressor = subprocess.Popen(
            ["zstd", "-q", "-T0", "-o", str(temporary)],
            stdin=subprocess.PIPE,
        )
        if compressor.stdin is None:
            raise RuntimeError("Failed to open zstd input")
        try:
            with tarfile.open(fileobj=compressor.stdin, mode="w|") as archive:
                encoded = json.dumps(metadata, sort_keys=True, indent=2).encode()
                info = _normalise_tar_info(tarfile.TarInfo("metadata.json"))
                info.size = len(encoded)
                archive.addfile(info, io.BytesIO(encoded))
                for path, arcname in files:
                    info = _normalise_tar_info(archive.gettarinfo(str(path), arcname=arcname))
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
        finally:
            compressor.stdin.close()
        if compressor.wait() != 0:
            raise RuntimeError("zstd failed while writing pyramid archive")


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
    config.run_dir.mkdir(parents=True, exist_ok=True)
    widths = range(config.chuda.min_width, config.chuda.max_width + 1)
    with tempfile.TemporaryDirectory(prefix="ansi-scaler-pyramids-", dir=config.run_dir) as temporary_name:
        temporary = Path(temporary_name)
        input_roots: dict[str, Path] = {}
        geometries: dict[str, dict[str, Any]] = {}
        prepared = []
        failures = 0
        for record in pending:
            output_id = pyramid_id(record, config)
            try:
                geometry = object_geometry(record, config)
                geometries[output_id] = geometry
                for width in (
                    config.chuda.min_width,
                    config.chuda.lod_3_below,
                    config.chuda.lod_2_below,
                    config.chuda.lod_1_below,
                ):
                    if not config.chuda.min_width <= width <= config.chuda.max_width:
                        continue
                    source_name, source_path = source_for_width(record, width, config)
                    root = input_roots.setdefault(source_name, temporary / "inputs" / source_name)
                    cropped = root / f"{output_id}.png"
                    if not cropped.exists():
                        _write_cropped_source(source_path, cropped, geometry["render_bbox"])
                prepared.append(record)
            except Exception as error:  # noqa: BLE001 - reject only the malformed image
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

        pending = prepared
        if not pending:
            return 0, failures, skipped

        rendered_root = temporary / "rendered"
        for width in tqdm(widths, desc="chuda widths", unit="width", dynamic_ncols=True):
            source_name, _ = source_for_width(pending[0], width, config)
            subprocess.run(
                [
                    config.chuda.executable,
                    str(input_roots[source_name]),
                    "--size",
                    str(width),
                    "--output",
                    str(rendered_root / f"{width:03d}"),
                ],
                check=True,
            )

        successes = 0
        for record in tqdm(pending, desc="pack pyramids", unit="pyramid", dynamic_ncols=True):
            output_id = pyramid_id(record, config)
            try:
                levels = []
                files = []
                for width in widths:
                    path = rendered_root / f"{width:03d}" / f"{output_id}.ansi"
                    data = path.read_bytes()
                    source_name, _ = source_for_width(record, width, config)
                    levels.append(
                        {
                            "width": width,
                            "rows": data.count(b"\n"),
                            "source_lod": source_name,
                            "bytes": len(data),
                            "sha256": sha256_file(path),
                            "path": f"levels/{width:03d}.ansi",
                        }
                    )
                    files.append((path, f"levels/{width:03d}.ansi"))
                metadata = {
                    "format": "ansi-scaler-pyramid-v1",
                    "id": output_id,
                    "parent_id": record["id"],
                    "chuda": config.chuda.model_dump(mode="json"),
                    "geometry": geometries[output_id],
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
                        "pyramid_format": metadata["format"],
                        "geometry": geometries[output_id],
                        "pyramid_levels": levels,
                    },
                )
                successes += 1
            except Exception as error:  # noqa: BLE001 - continue packaging independent pyramids
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
    return successes, failures, skipped
