from __future__ import annotations

import base64
import io
import json
import urllib.request
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from pydantic import BaseModel, Field

from ansi_scaler.config import RunConfig
from ansi_scaler.identity import stable_id
from ansi_scaler.manifests import read_jsonl, resolve_path
from ansi_scaler.reports import checkerboard
from ansi_scaler.runner import StageInfrastructureError, run_stage


CLASSIFIER_PROMPT = """Inspect this isolated game-art cutout rendered over a checkerboard.
Describe only what is visibly present. Do not infer the requested prompt or decide whether the image should be accepted.
Count separate candidate assets, not architectural parts or decorations. A collection intentionally joined into one object
is one asset; a sprite sheet or several alternative renderings contains multiple assets. Report visible incoherence,
unexplained geometry, residual background, damaged/missing regions, halos, stray fragments, text, or watermarks.
Use uncertainty when the primary object cannot be identified confidently."""


class Classification(BaseModel):
    description: str
    primary_object: str
    object_count: int = Field(ge=0)
    multiple_candidate_assets: bool
    visually_coherent: bool
    artifact_flags: list[str]
    uncertainty: float = Field(ge=0.0, le=1.0)


RequestFunction = Callable[[dict[str, Any]], dict[str, Any]]


class OllamaClassifier:
    def __init__(
        self,
        config: RunConfig,
        request_function: RequestFunction | None = None,
    ) -> None:
        self.config = config
        self.settings = config.vlm
        self.owns_request_function = request_function is None
        self.request_function = request_function or self._request
        self.used_model = False

    def output_id(self, source: dict[str, Any]) -> str:
        identity = {
            "model": self.settings.model,
            "prompt_version": self.settings.prompt_version,
            "temperature": self.settings.temperature,
        }
        return stable_id("vlm-classify-v1", source["id"], identity)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.settings.endpoint.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:  # noqa: S310
            return json.load(response)

    @staticmethod
    def _checkerboard_png(image: Image.Image) -> str:
        background = checkerboard(image.size, square=32)
        background.paste(image, mask=image.getchannel("A"))
        buffer = io.BytesIO()
        background.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def __call__(self, source: dict[str, Any]) -> dict[str, Any]:
        image = Image.open(resolve_path(source["artifact"], self.config.data_dir)).convert("RGBA")
        payload = {
            "model": self.settings.model,
            "stream": False,
            "format": Classification.model_json_schema(),
            "keep_alive": self.settings.keep_alive,
            "options": {"temperature": self.settings.temperature},
            "messages": [
                {
                    "role": "user",
                    "content": CLASSIFIER_PROMPT,
                    "images": [self._checkerboard_png(image)],
                }
            ],
        }
        try:
            response = self.request_function(payload)
        except OSError as error:
            raise StageInfrastructureError(
                f"VLM server {self.settings.endpoint} is unavailable; restore it and resume classification"
            ) from error
        self.used_model = True
        classification = Classification.model_validate_json(response["message"]["content"])
        return {
            **source,
            "id": self.output_id(source),
            "parent_id": source["id"],
            "stage": "classify",
            "classification": classification.model_dump(mode="json"),
            "vlm_model": self.settings.model,
            "vlm_prompt_version": self.settings.prompt_version,
            "eval_duration_ns": response.get("eval_duration"),
            "total_duration_ns": response.get("total_duration"),
        }

    def close(self) -> None:
        if not self.used_model or not self.owns_request_function:
            return
        payload = {"model": self.settings.model, "keep_alive": 0}
        request = urllib.request.Request(
            f"{self.settings.endpoint.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30):  # noqa: S310
                pass
        except OSError:
            pass


def run_classify(
    config: RunConfig,
    *,
    limit: int | None = None,
    artifact_ids: set[str] | None = None,
    force: bool = False,
    retry_errors: bool = False,
    request_function: RequestFunction | None = None,
) -> tuple[int, int, int]:
    processor = OllamaClassifier(config, request_function=request_function)
    inputs = read_jsonl(config.manifest_dir / "cutouts.jsonl")
    if artifact_ids:
        records = list(inputs)
        inputs = (
            record
            for record in records
            if record["id"] in artifact_ids or Path(record["artifact"]).stem in artifact_ids
        )
        found = {record["id"] for record in records if record["id"] in artifact_ids}
        found.update(
            Path(record["artifact"]).stem for record in records if Path(record["artifact"]).stem in artifact_ids
        )
        missing = artifact_ids - found
        if missing:
            raise ValueError(f"Unknown cutout artifact IDs: {', '.join(sorted(missing))}")
    return run_stage(
        inputs,
        config.manifest_dir / "classifications.jsonl",
        config.manifest_dir / "classifications.errors.jsonl",
        processor,
        processor.output_id,
        limit=limit or config.limit,
        force=force,
        retry_errors=retry_errors,
        stage_name=f"classify ({config.vlm.model})",
    )
