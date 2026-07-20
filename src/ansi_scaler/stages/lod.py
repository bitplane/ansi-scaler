from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cairosvg
import numpy as np
import psutil
import vtracer
from PIL import Image

from ansi_scaler.active import active_backgrounds
from ansi_scaler.artifacts import artifact_path, atomic_destination
from ansi_scaler.config import LodLevel, RunConfig
from ansi_scaler.identity import stable_id
from ansi_scaler.manifests import read_jsonl, relative_path, resolve_path
from ansi_scaler.reports import contact_sheet
from ansi_scaler.runner import run_parallel_stage


def prepare_trace_input(source: Path, destination: Path, alpha_threshold: int) -> None:
    with Image.open(source) as opened:
        pixels = np.array(opened.convert("RGBA"), copy=True)
    retained = pixels[:, :, 3] >= alpha_threshold
    pixels[~retained] = 0
    pixels[:, :, 3] = np.where(retained, 255, 0)
    Image.fromarray(pixels, mode="RGBA").save(destination, format="PNG")


def lod_worker_count(config: RunConfig) -> int:
    resources = config.resources
    cpu_count = psutil.cpu_count(logical=True) or 1
    if resources.lod_workers is not None:
        return min(cpu_count, resources.lod_workers)
    memory = psutil.virtual_memory()
    reserved = int(memory.total * resources.memory_headroom)
    usable = max(0, memory.available - reserved)
    per_worker = resources.lod_worker_memory_mb * 1024 * 1024
    return max(1, min(cpu_count, usable // per_worker))


class LodGenerator:
    def __init__(self, config: RunConfig, *, force: bool = False) -> None:
        self.config = config
        self.settings = config.lod
        self.force = force

    def output_id(self, source: dict[str, Any]) -> str:
        return stable_id("lod-v1", source["id"], self.settings.model_dump(mode="json"))

    def _generate_level(self, source_path: Path, output_id: str, level: LodLevel) -> dict[str, Any]:
        svg_id = stable_id(output_id, level.name, "svg")
        preview_id = stable_id(output_id, level.name, "preview")
        svg_path = artifact_path(self.config.artifact_dir, f"lod/{level.name}/svg", svg_id, ".svg")
        preview_path = artifact_path(self.config.artifact_dir, f"lod/{level.name}/preview", preview_id, ".png")
        if self.force:
            svg_path.unlink(missing_ok=True)
            preview_path.unlink(missing_ok=True)
        if not svg_path.exists():
            with atomic_destination(svg_path) as temporary:
                vtracer.convert_image_to_svg_py(
                    str(source_path),
                    str(temporary),
                    colormode="color",
                    hierarchical="stacked",
                    mode="spline",
                    filter_speckle=level.filter_speckle,
                    color_precision=level.color_precision,
                    layer_difference=level.layer_difference,
                    corner_threshold=level.corner_threshold,
                    length_threshold=level.length_threshold,
                    splice_threshold=level.splice_threshold,
                    path_precision=3,
                )
        if not preview_path.exists():
            with atomic_destination(preview_path) as temporary:
                cairosvg.svg2png(
                    url=str(svg_path),
                    write_to=str(temporary),
                    output_width=level.preview_size,
                    output_height=level.preview_size,
                )
        return {
            **level.model_dump(mode="json"),
            "svg": relative_path(svg_path, self.config.data_dir),
            "preview": relative_path(preview_path, self.config.data_dir),
            "svg_bytes": svg_path.stat().st_size,
        }

    def __call__(self, source: dict[str, Any]) -> dict[str, Any]:
        output_id = self.output_id(source)
        source_path = resolve_path(source["artifact"], self.config.data_dir)
        work_root = self.config.run_dir / "work" / "lod"
        work_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f"{output_id[:12]}-", dir=work_root) as temporary:
            trace_input = Path(temporary) / "trace-input.png"
            prepare_trace_input(source_path, trace_input, self.settings.alpha_threshold)
            levels = [self._generate_level(trace_input, output_id, level) for level in self.settings.levels]
        return {
            **source,
            "id": output_id,
            "parent_id": source["id"],
            "stage": "lod",
            "original": source["artifact"],
            "alpha_threshold": self.settings.alpha_threshold,
            "levels": levels,
        }


def run_lod(
    config: RunConfig,
    *,
    limit: int | None = None,
    force: bool = False,
    retry_errors: bool = False,
) -> tuple[int, int, int]:
    processor = LodGenerator(config, force=force)
    output = config.manifest_dir / "lods.jsonl"
    workers = lod_worker_count(config)
    result = run_parallel_stage(
        active_backgrounds(config),
        output,
        config.manifest_dir / "lods.errors.jsonl",
        processor,
        processor.output_id,
        workers=workers,
        limit=limit or config.limit,
        force=force,
        retry_errors=retry_errors,
        stage_name=f"lod ({workers} workers)",
    )
    active_ids = {processor.output_id(record) for record in active_backgrounds(config)}
    previews = [
        resolve_path(level["preview"], config.data_dir)
        for record in read_jsonl(output)
        if record["id"] in active_ids
        for level in record["levels"]
    ]
    contact_sheet(previews[:100], config.run_dir / "reports" / "lod-previews.png")
    return result
