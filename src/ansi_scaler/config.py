from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PromptSettings(BaseModel):
    template_config: Path = Path("configs/prompts/objects.yaml")
    variants_per_entry: int = Field(default=5, ge=1)
    seeds_per_prompt: int = Field(default=2, ge=1)
    base_seed: int = 42000


class SanaSettings(BaseModel):
    model_id: str = "Efficient-Large-Model/Sana_600M_512px_diffusers"
    revision: str = "2defc07f5fb66d0c53ace051585e9a2cb83f8c15"
    width: int = 512
    height: int = 512
    guidance_scale: float = 5.0
    inference_steps: int = 24
    device: str = "cuda"


class RembgSettings(BaseModel):
    model: str = "isnet-anime"
    sha256: str = "f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99"
    model_path: Path = Path("~/.u2net/isnet-anime.onnx")


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


class RunConfig(BaseModel):
    name: str
    catalog_dir: Path = Path("catalog")
    data_dir: Path = Path("data")
    limit: int | None = Field(default=None, ge=1)
    prompts: PromptSettings = PromptSettings()
    sana: SanaSettings = SanaSettings()
    rembg: RembgSettings = RembgSettings()
    lod: LodSettings
    resources: ResourceSettings = ResourceSettings()

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
