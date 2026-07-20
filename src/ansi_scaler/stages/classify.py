from __future__ import annotations

import base64
import io
import json
import urllib.request
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, Field

from ansi_scaler.active import active_backgrounds
from ansi_scaler.config import RunConfig
from ansi_scaler.identity import stable_id
from ansi_scaler.manifests import read_jsonl, resolve_path
from ansi_scaler.reports import checkerboard
from ansi_scaler.runner import run_stage
from ansi_scaler.stages.ollama import RequestFunction, request_with_retry


CLASSIFIER_PROMPT = """You are a factual visual observer for isolated game-art cutouts.
Describe only what is visibly present. Do not infer the requested prompt and do not decide whether the asset should be
accepted. The gray checkerboard is the viewer's visualization of transparent pixels: never describe it as an image
background, residual background, or artefact. Refer to it only as transparency when that fact matters.

A candidate asset is an independent alternative, duplicate, or sprite-sheet item. Held, mounted, attached, decorative,
or supporting parts belong in components and are not additional candidate assets. For example, a knight and held shield,
armour on a display stand, a drone with a mounted camera, grouped mushrooms, and a rock with attached moss are each one
candidate asset. List visible problems only; do not add boilerplate claims that text, damage, or other problems are absent.
Keep primary_subject as a short noun phrase. Put orientation, placement, pose, and part relationships in
spatial_description. Use low confidence and ambiguities when identification is genuinely unclear.

Return JSON matching this schema exactly:
{schema}"""


IssueKind = Literal[
    "incoherent_geometry",
    "missing_or_damaged_parts",
    "residual_background",
    "halo",
    "stray_fragment",
    "extraneous_object",
    "visible_text",
    "watermark",
    "cropped",
    "mostly_transparent",
    "unrecognizable",
]


class IssueObservation(BaseModel):
    kind: IssueKind
    evidence: str = Field(min_length=1)


class Classification(BaseModel):
    primary_subject: str = Field(min_length=1)
    description: str = Field(min_length=1)
    candidate_assets: list[str] = Field(min_length=1)
    components: list[str]
    spatial_description: str = Field(min_length=1)
    issues: list[IssueObservation]
    confidence: Literal["high", "medium", "low"]
    ambiguities: list[str]


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
            "num_predict": self.settings.num_predict,
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
            "think": False,
            "format": Classification.model_json_schema(),
            "keep_alive": self.settings.keep_alive,
            "options": {
                "temperature": self.settings.temperature,
                "num_predict": self.settings.num_predict,
            },
            "messages": [
                {
                    "role": "user",
                    "content": CLASSIFIER_PROMPT.format(
                        schema=json.dumps(Classification.model_json_schema(), sort_keys=True)
                    ),
                    "images": [self._checkerboard_png(image)],
                }
            ],
        }
        response = request_with_retry(
            self.request_function, payload, self.settings, service=f"VLM server {self.settings.endpoint}"
        )
        self.used_model = True
        try:
            classification = Classification.model_validate_json(response["message"]["content"])
        except Exception as error:
            raise ValueError(
                "VLM returned invalid structured output "
                f"(done_reason={response.get('done_reason')!r}, eval_count={response.get('eval_count')!r}, "
                f"total_duration_ns={response.get('total_duration')!r})"
            ) from error
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
            "prompt_eval_count": response.get("prompt_eval_count"),
            "eval_count": response.get("eval_count"),
            "done_reason": response.get("done_reason"),
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
    inputs = read_jsonl(config.manifest_dir / "backgrounds.jsonl") if artifact_ids else active_backgrounds(config)
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
