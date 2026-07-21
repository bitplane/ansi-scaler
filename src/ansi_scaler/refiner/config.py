from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field, model_validator

from ansi_scaler.config import load_yaml


class RefinerConfig(BaseModel):
    name: str
    dataset_recipe: Path
    output_root: Path = Path("data/training")
    seed: int = 42000
    device: str = "auto"
    steps: int = Field(default=20_000, ge=1)
    batch_size: int = Field(default=256, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_steps: int = Field(default=500, ge=0)
    checkpoint_steps: int = Field(default=1_000, ge=1)
    eval_steps: int = Field(default=500, ge=1)
    eval_patches_per_asset: int = Field(default=256, ge=1)
    d_model: int = Field(default=256, ge=32)
    heads: int = Field(default=4, ge=1)
    context_layers: int = Field(default=2, ge=1)
    decoder_layers: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    glyph_loss_weight: float = Field(default=1.0, ge=0)
    foreground_loss_weight: float = Field(default=2.0, ge=0)
    background_loss_weight: float = Field(default=2.0, ge=0)
    presence_loss_weight: float = Field(default=0.5, ge=0)
    train_asset_limit: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def valid_architecture(self) -> RefinerConfig:
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if self.warmup_steps > self.steps:
            raise ValueError("warmup_steps cannot exceed steps")
        return self


def load_refiner_config(path: Path) -> RefinerConfig:
    return RefinerConfig.model_validate(load_yaml(path))
