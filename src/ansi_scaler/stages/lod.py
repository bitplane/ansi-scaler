from __future__ import annotations

from pathlib import Path
from typing import Any

import cairosvg
import psutil
import vtracer

from ansi_scaler.artifacts import artifact_path, atomic_destination
from ansi_scaler.config import LodLevel, RunConfig
from ansi_scaler.identity import stable_id
from ansi_scaler.manifests import read_jsonl, relative_path, resolve_path
from ansi_scaler.reports import contact_sheet
from ansi_scaler.runner import run_parallel_stage


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
        levels = [self._generate_level(source_path, output_id, level) for level in self.settings.levels]
        return {
            **source,
            "id": output_id,
            "parent_id": source["id"],
            "stage": "lod",
            "original": source["artifact"],
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
        read_jsonl(config.manifest_dir / "cutouts.jsonl"),
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
    previews = [
        resolve_path(level["preview"], config.data_dir) for record in read_jsonl(output) for level in record["levels"]
    ]
    contact_sheet(previews[:100], config.run_dir / "reports" / "lod-previews.png")
    return result
