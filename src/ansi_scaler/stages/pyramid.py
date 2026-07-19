from __future__ import annotations

import errno
import importlib.metadata
import io
import json
import math
import shutil
import subprocess
import tarfile
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import cairosvg
from PIL import Image
from tqdm.auto import tqdm

from ansi_scaler.artifacts import artifact_path, atomic_destination
from ansi_scaler.config import RunConfig
from ansi_scaler.identity import sha256_file, stable_id
from ansi_scaler.manifests import append_jsonl, known_ids, read_jsonl, relative_path, resolve_path
from ansi_scaler.runner import StageInfrastructureError
from ansi_scaler.stages.lod import lod_worker_count


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


def _raise_if_infrastructure_error(error: Exception) -> None:
    infrastructure_errnos = {errno.EACCES, errno.ENOMEM, errno.ENOSPC, errno.EROFS}
    if isinstance(error, MemoryError) or (isinstance(error, OSError) and error.errno in infrastructure_errnos):
        raise StageInfrastructureError(f"Pyramid input preparation infrastructure failed: {error}") from error


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
    return stable_id(PYRAMID_FORMAT, record["id"], config.chuda.model_dump(mode="json"), rasterizer_signature())


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


def _save_shared_crop(
    image: Image.Image,
    destination: Path,
    canvas_size: list[int],
    render_bbox_px: list[int],
) -> None:
    if image.size != tuple(canvas_size):
        raise ValueError(f"Prepared source canvas mismatch: expected {tuple(canvas_size)}, got {image.size}")
    crop = image.convert("RGBA").crop(tuple(render_bbox_px))
    expected = (render_bbox_px[2] - render_bbox_px[0], render_bbox_px[3] - render_bbox_px[1])
    if crop.size != expected:
        raise ValueError(f"Prepared source crop mismatch: expected {expected}, got {crop.size}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    crop.save(destination, format="PNG")


def _write_cropped_original(
    source: Path,
    destination: Path,
    canvas_size: list[int],
    render_bbox_px: list[int],
) -> None:
    with Image.open(source) as image:
        _save_shared_crop(image, destination, canvas_size, render_bbox_px)


def _write_cropped_svg(
    source: Path,
    destination: Path,
    canvas_size: list[int],
    render_bbox_px: list[int],
) -> None:
    encoded = cairosvg.svg2png(
        url=str(source),
        output_width=canvas_size[0],
        output_height=canvas_size[1],
    )
    with Image.open(io.BytesIO(encoded)) as image:
        _save_shared_crop(image, destination, canvas_size, render_bbox_px)


def _prepare_record_inputs(
    record: dict[str, Any],
    config: RunConfig,
    temporary_root: str,
    source_names: tuple[str, ...],
) -> tuple[str, dict[str, Any]]:
    output_id = pyramid_id(record, config)
    geometry = object_geometry(record, config)
    destinations = []
    for source_name in source_names:
        if source_name == "original":
            source_path = resolve_path(record["original"], config.data_dir)
        else:
            level = next(level for level in record["levels"] if level["name"] == source_name)
            source_path = resolve_path(level["svg"], config.data_dir)
        destination = Path(temporary_root) / "inputs" / source_name / f"{output_id}.png"
        if source_name == "original":
            _write_cropped_original(
                source_path,
                destination,
                geometry["canvas_size"],
                geometry["render_bbox_px"],
            )
        else:
            _write_cropped_svg(
                source_path,
                destination,
                geometry["canvas_size"],
                geometry["render_bbox_px"],
            )
        destinations.append(destination)
    sizes = []
    for destination in destinations:
        with Image.open(destination) as image:
            sizes.append(image.size)
    expected = tuple(geometry["render_size_px"])
    if any(size != expected for size in sizes):
        raise ValueError(f"Prepared pyramid inputs do not share crop size {expected}: {sizes}")
    return output_id, geometry


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
        source_names = tuple(
            dict.fromkeys(source_for_width(pending[0], width, config)[0] for width in widths)
        )
        input_roots = {source_name: temporary / "inputs" / source_name for source_name in source_names}
        geometries: dict[str, dict[str, Any]] = {}
        prepared = []
        failures = 0
        workers = min(lod_worker_count(config), len(pending))
        progress = tqdm(total=len(pending), desc="prepare pyramid inputs", unit="image", dynamic_ncols=True)
        progress.set_postfix(completed=0, failed=0, skipped=skipped, refresh=False)

        def accept_result(record: dict[str, Any], result: tuple[str, dict[str, Any]]) -> None:
            output_id, geometry = result
            geometries[output_id] = geometry
            prepared.append(record)

        def reject_record(record: dict[str, Any], error: Exception) -> None:
            nonlocal failures
            append_jsonl(
                error_manifest,
                {
                    "parent_id": record["id"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": "".join(traceback.format_exception(error)),
                },
            )
            failures += 1

        try:
            if workers <= 1:
                for record in pending:
                    try:
                        accept_result(
                            record,
                            _prepare_record_inputs(record, config, temporary_name, source_names),
                        )
                    except Exception as error:  # noqa: BLE001 - reject only the malformed image
                        _raise_if_infrastructure_error(error)
                        reject_record(record, error)
                    progress.update()
                    progress.set_postfix(completed=len(prepared), failed=failures, skipped=skipped, refresh=False)
            else:
                with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
                    futures = {
                        executor.submit(_prepare_record_inputs, record, config, temporary_name, source_names): record
                        for record in pending
                    }
                    for future in as_completed(futures):
                        record = futures[future]
                        try:
                            accept_result(record, future.result())
                        except (BrokenProcessPool, BrokenPipeError, EOFError) as error:
                            raise StageInfrastructureError("Pyramid input worker pool failed") from error
                        except StageInfrastructureError:
                            raise
                        except Exception as error:  # noqa: BLE001 - reject only the malformed image
                            _raise_if_infrastructure_error(error)
                            reject_record(record, error)
                        progress.update()
                        progress.set_postfix(
                            completed=len(prepared), failed=failures, skipped=skipped, refresh=False
                        )
        finally:
            progress.close()

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
                    "format": PYRAMID_FORMAT,
                    "id": output_id,
                    "parent_id": record["id"],
                    "chuda": config.chuda.model_dump(mode="json"),
                    "input_rasterization": rasterizer_signature(),
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
                        "input_rasterization": metadata["input_rasterization"],
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
