from __future__ import annotations

import gc
from typing import Any

from PIL import Image

from ansi_scaler.artifacts import artifact_path, atomic_destination
from ansi_scaler.config import RunConfig
from ansi_scaler.identity import stable_id
from ansi_scaler.manifests import read_jsonl, relative_path, resolve_path
from ansi_scaler.reports import contact_sheet
from ansi_scaler.runner import StageInfrastructureError, run_stage


class SanaGenerator:
    def __init__(self, config: RunConfig, pipeline: Any | None = None, *, force: bool = False) -> None:
        self.config = config
        self.settings = config.sana
        self.pipeline = pipeline
        self.force = force
        if self.pipeline is not None:
            self._configure_pipeline()

    def _configure_pipeline(self) -> None:
        configure = getattr(self.pipeline, "set_progress_bar_config", None)
        if callable(configure):
            configure(disable=True)

    def _load_pipeline(self) -> Any:
        if self.pipeline is None:
            import torch
            from diffusers import SanaPipeline

            if self.settings.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("Sana generation is configured for CUDA, but CUDA is unavailable")
            if self.settings.device == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32
            load_options = {"revision": self.settings.revision, "torch_dtype": dtype}
            if self.settings.variant:
                load_options["variant"] = self.settings.variant
            self.pipeline = SanaPipeline.from_pretrained(self.settings.model_id, **load_options)
            self.pipeline.to(self.settings.device)
            self._configure_pipeline()
        return self.pipeline

    def close(self) -> None:
        had_pipeline = self.pipeline is not None
        self.pipeline = None
        gc.collect()
        if had_pipeline and self.settings.device == "cuda":
            import torch

            torch.cuda.empty_cache()

    def output_id(self, source: dict[str, Any]) -> str:
        return stable_id("sana-v1", source["id"], self.settings.model_dump(mode="json"))

    def __call__(self, source: dict[str, Any]) -> dict[str, Any]:
        import torch

        output_id = self.output_id(source)
        destination = artifact_path(self.config.artifact_dir, "rasters", output_id, ".png")
        if self.force:
            destination.unlink(missing_ok=True)
        if not destination.exists():
            generator = torch.Generator(device=self.settings.device).manual_seed(source["seed"])
            try:
                image: Image.Image = self._load_pipeline()(
                    prompt=source["prompt"],
                    negative_prompt=source["negative_prompt"],
                    height=self.settings.height,
                    width=self.settings.width,
                    guidance_scale=self.settings.guidance_scale,
                    num_inference_steps=self.settings.inference_steps,
                    generator=generator,
                ).images[0]
            except Exception as error:
                raise StageInfrastructureError(
                    "Sana could not initialise or run on this machine; fix the reported dependency/CUDA error and resume"
                ) from error
            with atomic_destination(destination) as temporary:
                image.save(temporary, format="PNG")
        return {
            **source,
            "id": output_id,
            "parent_id": source["id"],
            "stage": "generate",
            "artifact": relative_path(destination, self.config.data_dir),
            "generator_model": self.settings.model_id,
            "generator_revision": self.settings.revision,
            "generator_settings": self.settings.model_dump(mode="json"),
        }


def run_generate(
    config: RunConfig,
    *,
    limit: int | None = None,
    force: bool = False,
    retry_errors: bool = False,
    pipeline: Any | None = None,
) -> tuple[int, int, int]:
    processor = SanaGenerator(config, pipeline=pipeline, force=force)
    output = config.manifest_dir / "rasters.jsonl"
    result = run_stage(
        read_jsonl(config.manifest_dir / "prompts.jsonl"),
        output,
        config.manifest_dir / "rasters.errors.jsonl",
        processor,
        processor.output_id,
        limit=limit or config.limit,
        force=force,
        retry_errors=retry_errors,
        stage_name="generate",
    )
    paths = [resolve_path(record["artifact"], config.data_dir) for record in read_jsonl(output)]
    contact_sheet(paths[:100], config.run_dir / "reports" / "rasters.png")
    return result
