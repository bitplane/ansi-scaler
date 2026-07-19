from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import tarfile
import traceback
from concurrent.futures import FIRST_COMPLETED, BrokenExecutor, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import contextmanager
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import cairosvg
import chuda
import psutil
import zstandard
from PIL import Image
from tqdm.auto import tqdm

from ansi_scaler.artifacts import artifact_path, atomic_destination
from ansi_scaler.config import RunConfig
from ansi_scaler.identity import sha256_file, stable_id
from ansi_scaler.manifests import append_jsonl, known_ids, read_jsonl, relative_path, resolve_path
from ansi_scaler.runner import StageInfrastructureError


PYRAMID_FORMAT = "ansi-scaler-pyramid-v3"
INPUT_RASTERIZATION_CONTRACT = "shared-crop-premultiplied-lod-blend-v2"
LOD_BLEND_RADIUS = 4


@lru_cache(maxsize=1)
def rasterizer_signature() -> dict[str, str]:
    return {
        "contract": INPUT_RASTERIZATION_CONTRACT,
        "cairosvg_version": importlib.metadata.version("cairosvg"),
        "pillow_version": importlib.metadata.version("pillow"),
        "mode": "RGBA",
    }


def source_for_width(record: dict[str, Any], width: int, config: RunConfig) -> tuple[tuple[str, Path, float], ...]:
    settings = config.chuda
    transitions = (
        ("lod-3", "lod-2", settings.lod_3_below),
        ("lod-2", "lod-1", settings.lod_2_below),
        ("lod-1", "lod-0", settings.lod_1_below),
    )

    def source(name: str, weight: float) -> tuple[str, Path, float]:
        level = next(level for level in record["levels"] if level["name"] == name)
        return name, resolve_path(level["svg"], config.data_dir), weight

    for lower, higher, boundary in transitions:
        start = boundary - LOD_BLEND_RADIUS
        end = boundary + LOD_BLEND_RADIUS
        if start < width < end:
            higher_weight = (width - start) / (2 * LOD_BLEND_RADIUS)
            return source(lower, 1.0 - higher_weight), source(higher, higher_weight)
    if width < settings.lod_3_below:
        level_name = "lod-3"
    elif width < settings.lod_2_below:
        level_name = "lod-2"
    elif width < settings.lod_1_below:
        level_name = "lod-1"
    else:
        level_name = "lod-0"
    return (source(level_name, 1.0),)


def _source_key(width: int, sources: tuple[tuple[str, Path, float], ...]) -> str:
    return sources[0][0] if len(sources) == 1 else f"blend-{width:03d}"


def _source_provenance(sources: tuple[tuple[str, Path, float], ...]) -> list[dict[str, Any]]:
    return [{"name": name, "weight": weight} for name, _path, weight in sources]


def _blend_rgba_buffers(
    lower: tuple[int, int, bytes], higher: tuple[int, int, bytes], higher_weight: float
) -> tuple[int, int, bytes]:
    lower_width, lower_height, lower_data = lower
    higher_width, higher_height, higher_data = higher
    if (lower_width, lower_height) != (higher_width, higher_height):
        raise ValueError("LOD blend sources do not share identical dimensions")
    lower_image = Image.frombytes("RGBA", (lower_width, lower_height), lower_data).convert("RGBa")
    higher_image = Image.frombytes("RGBA", (higher_width, higher_height), higher_data).convert("RGBa")
    blended = Image.blend(lower_image, higher_image, higher_weight).convert("RGBA")
    return blended.width, blended.height, blended.tobytes()


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


def pyramid_worker_count(config: RunConfig) -> int:
    resources = config.resources
    cpu_count = psutil.cpu_count(logical=True) or 1
    if resources.pyramid_workers is not None:
        return min(cpu_count, resources.pyramid_workers)
    memory = psutil.virtual_memory()
    reserved = int(memory.total * resources.memory_headroom)
    usable = max(0, memory.available - reserved)
    per_worker = resources.pyramid_worker_memory_mb * 1024 * 1024
    memory_workers = usable // per_worker
    cpu_workers = max(1, cpu_count - 2)
    return max(1, min(cpu_workers, memory_workers))


def pyramid_queue_limits(workers: int) -> tuple[int, int]:
    return workers + 1, 2


def _crop_record_source_buffers(
    record: dict[str, Any], config: RunConfig, geometry: dict[str, Any], source_names: tuple[str, ...]
) -> dict[str, tuple[int, int, bytes]]:
    canvas_size = geometry["canvas_size"]
    render_bbox = tuple(geometry["render_bbox_px"])
    result: dict[str, tuple[int, int, bytes]] = {}
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
        result[source_name] = (cropped.width, cropped.height, cropped.tobytes())
    return result


def _crop_record_sources(
    record: dict[str, Any], config: RunConfig, geometry: dict[str, Any], source_names: tuple[str, ...]
) -> dict[str, chuda.Image]:
    return {
        name: chuda.Image.from_rgba(width, height, data)
        for name, (width, height, data) in _crop_record_source_buffers(record, config, geometry, source_names).items()
    }


def _prepare_record(
    record: dict[str, Any], config: RunConfig, widths: tuple[int, ...]
) -> tuple[dict[str, Any], dict[str, tuple[int, int, bytes]], float]:
    started = perf_counter()
    geometry = object_geometry(record, config)
    mixes = {width: source_for_width(record, width, config) for width in widths}
    source_names = tuple(dict.fromkeys(source[0] for sources in mixes.values() for source in sources))
    sources = _crop_record_source_buffers(record, config, geometry, source_names)
    for width, mix in mixes.items():
        if len(mix) == 1:
            continue
        sources[_source_key(width, mix)] = _blend_rgba_buffers(sources[mix[0][0]], sources[mix[1][0]], mix[1][2])
    return geometry, sources, perf_counter() - started


def _pack_archive(destination: Path, metadata: dict[str, Any], files: list[tuple[bytes, str]]) -> tuple[str, float]:
    started = perf_counter()
    _write_archive(destination, metadata, files)
    return sha256_file(destination), perf_counter() - started


@contextmanager
def _pipeline_executors(
    workers: int,
    prepare_futures: dict[Future[Any], dict[str, Any]],
    pack_futures: dict[Future[Any], tuple[dict[str, Any], int]],
) -> Iterator[tuple[ProcessPoolExecutor, ThreadPoolExecutor]]:
    with (
        ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as prepare_executor,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="pyramid-pack") as pack_executor,
    ):
        try:
            yield prepare_executor, pack_executor
        except BaseException:
            for future in (*prepare_futures, *pack_futures):
                future.cancel()
            raise


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
    boundaries = (config.chuda.lod_3_below, config.chuda.lod_2_below, config.chuda.lod_1_below)
    if any(higher - lower < 2 * LOD_BLEND_RADIUS for lower, higher in zip(boundaries, boundaries[1:])):
        raise ValueError("Chuda LOD boundaries are too close for the fixed blend radius")
    output_manifest = config.manifest_dir / "pyramids.jsonl"
    error_manifest = config.manifest_dir / "pyramids.errors.jsonl"
    if force:
        output_manifest.unlink(missing_ok=True)
        error_manifest.unlink(missing_ok=True)
    completed = known_ids(output_manifest)
    errors = list(read_jsonl(error_manifest)) if not retry_errors else []
    failed_outputs = {record["output_id"] for record in errors if record.get("output_id")}
    legacy_failed_parents = {record["parent_id"] for record in errors if not record.get("output_id")}
    unique_records = {record["id"]: record for record in read_jsonl(config.manifest_dir / "lods.jsonl")}
    selected = list(unique_records.values())[: limit or config.limit]
    pending = [
        record
        for record in selected
        if pyramid_id(record, config) not in completed
        and pyramid_id(record, config) not in failed_outputs
        and record["id"] not in legacy_failed_parents
    ]
    skipped = len(selected) - len(pending)
    if not pending:
        return 0, 0, skipped

    _check_chuda(config)
    widths = tuple(range(config.chuda.min_width, config.chuda.max_width + 1))
    renderer = chuda.Renderer(config.chuda.backend, config.chuda.max_batch_cells)
    workers = min(pyramid_worker_count(config), len(pending))
    prepare_limit, pack_limit = pyramid_queue_limits(workers)
    successes = 0
    failures = 0
    rendered_cells = 0
    timings = {"prepare": 0.0, "render": 0.0, "pack": 0.0}
    timing_counts = {"prepare": 0, "render": 0, "pack": 0}
    progress = tqdm(total=len(pending), desc=f"pyramids ({workers} prep workers)", unit="pyramid", dynamic_ncols=True)

    def update_progress(prepare_queued: int, pack_queued: int) -> None:
        averages = {
            name: round(timings[name] * 1000 / timing_counts[name]) if timing_counts[name] else 0 for name in timings
        }
        progress.set_postfix(
            completed=successes,
            failed=failures,
            skipped=skipped,
            cells=rendered_cells,
            prep_ms=averages["prepare"],
            render_ms=averages["render"],
            pack_ms=averages["pack"],
            queued=f"{prepare_queued}/{pack_queued}",
            refresh=False,
        )

    def fail_record(record: dict[str, Any], error: Exception) -> None:
        nonlocal failures
        append_jsonl(
            error_manifest,
            {
                "parent_id": record["id"],
                "output_id": pyramid_id(record, config),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        failures += 1
        progress.update()

    update_progress(0, 0)
    prepare_futures: dict[Future[Any], dict[str, Any]] = {}
    pack_futures: dict[Future[Any], tuple[dict[str, Any], int]] = {}
    try:
        with _pipeline_executors(workers, prepare_futures, pack_futures) as (prepare_executor, pack_executor):
            remaining = iter(pending)

            def submit_preparation() -> None:
                while len(prepare_futures) < prepare_limit:
                    record = next(remaining, None)
                    if record is None:
                        break
                    try:
                        future = prepare_executor.submit(_prepare_record, record, config, widths)
                    except BrokenExecutor as error:
                        raise StageInfrastructureError(f"Pyramid preparation pool failed: {error}") from error
                    prepare_futures[future] = record

            def finish_packs(*, block: bool) -> None:
                nonlocal successes, rendered_cells
                if not pack_futures:
                    return
                if block:
                    done, _ = wait(pack_futures, return_when=FIRST_COMPLETED)
                else:
                    done = {future for future in pack_futures if future.done()}
                for future in done:
                    output_record, cells = pack_futures.pop(future)
                    try:
                        archive_sha256, elapsed = future.result()
                        append_jsonl(output_manifest, {**output_record, "archive_sha256": archive_sha256})
                    except Exception as error:  # noqa: BLE001 - archive failures are stage-wide
                        raise StageInfrastructureError(f"Pyramid archive infrastructure failed: {error}") from error
                    timings["pack"] += elapsed
                    timing_counts["pack"] += 1
                    successes += 1
                    rendered_cells += cells
                    progress.update()
                    update_progress(len(prepare_futures), len(pack_futures))

            submit_preparation()
            while prepare_futures:
                prepared, _ = wait(prepare_futures, return_when=FIRST_COMPLETED)
                for future in prepared:
                    record = prepare_futures.pop(future)
                    output_id = pyramid_id(record, config)
                    try:
                        geometry, source_buffers, prepare_elapsed = future.result()
                        timings["prepare"] += prepare_elapsed
                        timing_counts["prepare"] += 1
                        sources = {
                            name: chuda.Image.from_rgba(width, height, data)
                            for name, (width, height, data) in source_buffers.items()
                        }
                        width_sources = {width: source_for_width(record, width, config) for width in widths}
                        requests = [(sources[_source_key(width, width_sources[width])], width) for width in widths]
                        render_started = perf_counter()
                        frames = renderer.render_many(
                            requests,
                            font_ratio=config.chuda.font_ratio,
                            transparent_threshold=config.chuda.transparent_threshold,
                        )
                        levels = []
                        files = []
                        backends = set()
                        record_cells = 0
                        for width, frame in zip(widths, frames, strict=True):
                            data = bytes(frame.to_ansi())
                            source_mix = width_sources[width]
                            source_lods = _source_provenance(source_mix)
                            path = f"levels/{width:03d}.ansi"
                            levels.append(
                                {
                                    "width": width,
                                    "rows": frame.rows,
                                    "source_lod": "/".join(source["name"] for source in source_lods),
                                    "source_lods": source_lods,
                                    "bytes": len(data),
                                    "sha256": hashlib.sha256(data).hexdigest(),
                                    "path": path,
                                }
                            )
                            files.append((data, path))
                            backends.add(frame.backend)
                            record_cells += frame.columns * frame.rows
                        timings["render"] += perf_counter() - render_started
                        timing_counts["render"] += 1
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
                        while len(pack_futures) >= pack_limit:
                            finish_packs(block=True)
                        output_record = {
                            **record,
                            "id": output_id,
                            "parent_id": record["id"],
                            "stage": "pyramid",
                            "artifact": relative_path(destination, config.data_dir),
                            "chuda_version": config.chuda.version,
                            "chuda_backends": metadata["chuda_backends"],
                            "pyramid_format": metadata["format"],
                            "input_rasterization": metadata["input_rasterization"],
                            "geometry": geometry,
                            "pyramid_levels": levels,
                        }
                        pack_future = pack_executor.submit(_pack_archive, destination, metadata, files)
                        pack_futures[pack_future] = (output_record, record_cells)
                        finish_packs(block=False)
                    except BrokenExecutor as error:
                        raise StageInfrastructureError(f"Pyramid preparation pool failed: {error}") from error
                    except (MemoryError, OSError) as error:
                        raise StageInfrastructureError(f"Pyramid infrastructure failed: {error}") from error
                    except StageInfrastructureError:
                        raise
                    except RuntimeError as error:
                        if config.chuda.backend == "cuda":
                            raise StageInfrastructureError(f"Chuda CUDA backend failed: {error}") from error
                        fail_record(record, error)
                    except Exception as error:  # noqa: BLE001 - reject only the malformed source record
                        fail_record(record, error)
                    submit_preparation()
                    update_progress(len(prepare_futures), len(pack_futures))

            while pack_futures:
                finish_packs(block=True)
    finally:
        progress.close()
    return successes, failures, skipped
