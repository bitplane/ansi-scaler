from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ansi_scaler.config import load_yaml


class SplitSettings(BaseModel):
    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    @model_validator(mode="after")
    def valid_total(self) -> SplitSettings:
        if abs(self.train + self.validation + self.test - 1.0) > 1e-9:
            raise ValueError("Dataset split fractions must sum to one")
        return self


class DatasetRecipe(BaseModel):
    name: str
    run_config: Path
    format: Literal["ansi-pyramid-tensors-v1"] = "ansi-pyramid-tensors-v1"
    selection_policy: Literal["human-override-verifier-accept-v1"] = "human-override-verifier-accept-v1"
    pyramid_format: str = "ansi-scaler-pyramid-v3"
    split_seed: int = 42000
    splits: SplitSettings = SplitSettings()
    shard_count: int = Field(default=16, ge=1, le=1024)
    base_vocabulary: Path | None = None
    output_root: Path = Path("data/datasets")


def load_dataset_recipe(path: Path) -> DatasetRecipe:
    return DatasetRecipe.model_validate(load_yaml(path))
