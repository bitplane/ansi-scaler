from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class PromptSettings(BaseModel):
    seeds_per_specification: int = Field(default=2, ge=1)
    base_seed: int = 42000


class SanaSettings(BaseModel):
    model_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    revision: str = "e58e81ec2dc4faf305122872313884df3afeffa8"
    variant: str | None = "fp16"
    width: int = 512
    height: int = 512
    guidance_scale: float = 4.5
    inference_steps: int = 20
    device: str = "cuda"
    presentation_prompt: str = (
        "single isolated subject, fully visible with comfortable margin, centered, polished cartoon game asset, "
        "strong readable silhouette, chunky simple shapes, clean crisp edges, vibrant controlled color palette, "
        "flat light-gray studio background, even soft lighting, minimal contact shadow, no scenery, no surrounding "
        "props, no border, no frame, no text, no logo"
    )
    exclusions: list[str] = Field(
        default_factory=lambda: [
            "multiple subjects",
            "duplicate object",
            "cropped",
            "cut off",
            "complex background",
            "landscape",
            "room",
            "scenery",
            "pedestal",
            "decorative border",
            "text",
            "letters",
            "logo",
            "watermark",
            "photorealistic",
        ]
    )


class BackgroundSettings(BaseModel):
    provider: Literal["rembg-onnx", "lucida-transformers"] = "lucida-transformers"
    model: str = "egeorcun/lucida"
    revision: str | None = "28632b8fefc5431cfc1e42ed9d6123d785ea49ad"
    sha256: str | None = None
    model_path: Path | None = None
    input_size: int = Field(default=1024, ge=64)
    device: str = "cuda"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"

    @model_validator(mode="after")
    def validate_provider_settings(self) -> BackgroundSettings:
        if self.provider == "lucida-transformers":
            if not self.revision:
                raise ValueError("lucida-transformers requires a pinned revision")
            self.sha256 = None
            self.model_path = None
        elif self.sha256 is None or self.model_path is None:
            raise ValueError("rembg-onnx requires sha256 and model_path")
        return self


class LodLevel(BaseModel):
    name: str
    preview_size: int
    min_cells: int
    max_cells: int
    filter_speckle: int
    color_precision: int
    layer_difference: int
    length_threshold: float
    corner_threshold: int
    splice_threshold: int


class LodSettings(BaseModel):
    levels: list[LodLevel]


class ResourceSettings(BaseModel):
    memory_headroom: float = Field(default=0.2, ge=0.0, lt=1.0)
    lod_worker_memory_mb: int = Field(default=256, ge=1)
    lod_workers: int | None = Field(default=None, ge=1)
    pyramid_worker_memory_mb: int = Field(default=128, ge=1)
    pyramid_workers: int | None = Field(default=None, ge=1)


class VlmSettings(BaseModel):
    model: str = "qwen3-vl:8b"
    endpoint: str = "http://127.0.0.1:11434"
    prompt_version: str = "cutout-classifier-v1"
    temperature: float = Field(default=0.0, ge=0.0)
    timeout_seconds: int = Field(default=300, ge=1)
    keep_alive: str = "10m"
    retry_attempts: int = Field(default=4, ge=1)
    retry_initial_seconds: float = Field(default=1.0, ge=0.0)
    retry_max_seconds: float = Field(default=8.0, ge=0.0)


class LlmSettings(BaseModel):
    model: str = "qwen3:8b"
    endpoint: str = "http://127.0.0.1:11434"
    prompt_version: str = "corpus-verifier-v2"
    temperature: float = Field(default=0.0, ge=0.0)
    timeout_seconds: int = Field(default=300, ge=1)
    keep_alive: str = "10m"
    retry_attempts: int = Field(default=4, ge=1)
    retry_initial_seconds: float = Field(default=1.0, ge=0.0)
    retry_max_seconds: float = Field(default=8.0, ge=0.0)


class ChudaSettings(BaseModel):
    version: str = "0.2.3"
    backend: str = Field(default="auto", pattern="^(auto|cpu|cuda)$")
    max_batch_cells: int = Field(default=262_144, ge=1)
    font_ratio: float = Field(default=2.0, gt=0.0)
    transparent_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    min_width: int = Field(default=2, ge=1)
    max_width: int = Field(default=120, ge=1)
    lod_3_below: int = Field(default=10, ge=1)
    lod_2_below: int = Field(default=40, ge=1)
    lod_1_below: int = Field(default=80, ge=1)
    alpha_bbox_threshold: int = Field(default=1, ge=0, le=255)
    crop_padding_fraction: float = Field(default=0.05, ge=0.0, le=1.0)


class ReviewIssueSettings(BaseModel):
    label: str
    default_stage: str


class ReviewSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    refresh_seconds: float = Field(default=2.0, ge=0.25)
    issues: dict[str, ReviewIssueSettings] = Field(
        default_factory=lambda: {
            "wrong_subject": ReviewIssueSettings(label="Wrong subject", default_stage="generate"),
            "multiple_assets": ReviewIssueSettings(label="Multiple assets", default_stage="generate"),
            "incoherent_geometry": ReviewIssueSettings(label="Incoherent geometry", default_stage="generate"),
            "cropped_source": ReviewIssueSettings(label="Source is cropped", default_stage="generate"),
            "missing_parts": ReviewIssueSettings(label="Missing parts", default_stage="background"),
            "mostly_transparent": ReviewIssueSettings(label="Mostly transparent", default_stage="background"),
            "residual_background": ReviewIssueSettings(label="Residual background", default_stage="background"),
            "halo": ReviewIssueSettings(label="Background halo", default_stage="background"),
            "stray_fragments": ReviewIssueSettings(label="Stray fragments", default_stage="background"),
            "lod_detail_loss": ReviewIssueSettings(label="LOD damage", default_stage="lod"),
            "ansi_render_failure": ReviewIssueSettings(label="ANSI damage", default_stage="pyramid"),
            "prompt_problem": ReviewIssueSettings(label="Prompt problem", default_stage="prompt"),
            "classifier_error": ReviewIssueSettings(label="VLM observation is wrong", default_stage="classify"),
            "verifier_error": ReviewIssueSettings(label="Verifier decision is wrong", default_stage="verify"),
            "other": ReviewIssueSettings(label="Other", default_stage="generate"),
        }
    )


class RunConfig(BaseModel):
    name: str
    content_dir: Path = Path("content")
    data_dir: Path = Path("data")
    limit: int | None = Field(default=None, ge=1)
    prompts: PromptSettings = PromptSettings()
    sana: SanaSettings = SanaSettings()
    background: BackgroundSettings = BackgroundSettings()
    lod: LodSettings
    resources: ResourceSettings = ResourceSettings()
    vlm: VlmSettings = VlmSettings()
    llm: LlmSettings = LlmSettings()
    chuda: ChudaSettings = ChudaSettings()
    review: ReviewSettings = ReviewSettings()

    @property
    def run_dir(self) -> Path:
        return self.data_dir / "runs" / self.name

    @property
    def manifest_dir(self) -> Path:
        return self.run_dir / "manifests"

    @property
    def artifact_dir(self) -> Path:
        return self.data_dir / "artifacts"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def load_run_config(path: Path) -> RunConfig:
    return RunConfig.model_validate(load_yaml(path))
