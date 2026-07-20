from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, model_validator

from ansi_scaler.active import active_classifications
from ansi_scaler.config import RunConfig
from ansi_scaler.identity import stable_id
from ansi_scaler.manifests import read_jsonl
from ansi_scaler.runner import run_stage
from ansi_scaler.stages.ollama import OllamaStructuredOutputError, RequestFunction, request_with_retry


VERIFIER_PROMPT = """You are a conservative quality gate for a synthetic game-art corpus. Compare the requested semantic
concept with an independent vision model's factual observations. Judge semantic equivalence rather than exact noun
matching: synonyms and visible held, mounted, attached, or supporting components can satisfy the concept. The checkerboard
is a visualization of transparency and is never a background mismatch. Do not judge studio presentation, style scaffolding,
or other generator instructions that are not present in the semantic request.

Reject clear wrong subjects, genuinely separate candidate alternatives, incoherent generation, or cutout damage that makes
the asset unusable. Attribute wrong content or incoherent source imagery to generate; attribute missing regions, halos,
residual backgrounds, stray cutout fragments, or excessive transparency to background. If evidence is ambiguous or
insufficient choose review, never accept. Return JSON matching the supplied schema only."""


Judgment = Literal["match", "mismatch", "uncertain"]
QualityJudgment = Literal["usable", "unusable", "uncertain"]


class Verification(BaseModel):
    semantic_match: Judgment
    cardinality_match: Judgment
    quality: QualityJudgment
    decision: Literal["accept", "reject", "review"]
    failed_stage: Literal["generate", "background"] | None
    reasons: list[str]
    explanation: str

    @model_validator(mode="after")
    def validate_decision(self) -> Verification:
        judgments = (self.semantic_match, self.cardinality_match, self.quality)
        if self.decision == "accept":
            if judgments != ("match", "match", "usable") or self.failed_stage is not None:
                raise ValueError("Accepted verification must be an unambiguously usable match")
        elif self.decision == "reject":
            if "mismatch" not in judgments and "unusable" not in judgments:
                raise ValueError("Rejected verification requires a mismatch or unusable result")
            if self.failed_stage is None:
                raise ValueError("Rejected verification requires a failed stage")
        elif "uncertain" not in judgments:
            raise ValueError("Review verification requires uncertain evidence")
        return self


class OllamaVerifier:
    def __init__(self, config: RunConfig, request_function: RequestFunction | None = None) -> None:
        self.settings = config.llm
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
        return stable_id("llm-verify-v1", source["id"], identity)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.settings.endpoint.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:  # noqa: S310
            return json.load(response)

    def __call__(self, source: dict[str, Any]) -> dict[str, Any]:
        evidence = {
            "requested_concept": source.get("label", source.get("concept_name")),
            "requested_concept_id": source.get("specification_id", source.get("concept_id")),
            "semantic_prompt": source.get("semantic_prompt", source.get("label", source.get("concept_name"))),
            "vision_observations": {
                key: value for key, value in source["classification"].items() if key != "spatial_description"
            },
        }
        payload = {
            "model": self.settings.model,
            "stream": False,
            "think": False,
            "format": Verification.model_json_schema(),
            "keep_alive": self.settings.keep_alive,
            "options": {
                "temperature": self.settings.temperature,
                "num_predict": self.settings.num_predict,
            },
            "messages": [
                {
                    "role": "system",
                    "content": f"{VERIFIER_PROMPT}\n\nSchema:\n{json.dumps(Verification.model_json_schema(), sort_keys=True)}",
                },
                {"role": "user", "content": json.dumps(evidence, sort_keys=True)},
            ],
        }
        response = request_with_retry(
            self.request_function, payload, self.settings, service=f"LLM server {self.settings.endpoint}"
        )
        self.used_model = True
        try:
            verification = Verification.model_validate_json(response["message"]["content"])
        except Exception as error:
            raise OllamaStructuredOutputError("LLM", response) from error
        return {
            **source,
            "id": self.output_id(source),
            "parent_id": source["id"],
            "stage": "verify",
            "verification": verification.model_dump(mode="json"),
            "llm_model": self.settings.model,
            "llm_prompt_version": self.settings.prompt_version,
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


def run_verify(
    config: RunConfig,
    *,
    limit: int | None = None,
    artifact_ids: set[str] | None = None,
    force: bool = False,
    retry_errors: bool = False,
    request_function: RequestFunction | None = None,
) -> tuple[int, int, int]:
    processor = OllamaVerifier(config, request_function=request_function)
    inputs = (
        read_jsonl(config.manifest_dir / "classifications.jsonl") if artifact_ids else active_classifications(config)
    )
    if artifact_ids:
        records = list(inputs)
        inputs = (record for record in records if Path(record["artifact"]).stem in artifact_ids)
        found = {Path(record["artifact"]).stem for record in records if Path(record["artifact"]).stem in artifact_ids}
        missing = artifact_ids - found
        if missing:
            raise ValueError(f"Unknown classified artifact IDs: {', '.join(sorted(missing))}")
    return run_stage(
        inputs,
        config.manifest_dir / "verifications.jsonl",
        config.manifest_dir / "verifications.errors.jsonl",
        processor,
        processor.output_id,
        limit=limit or config.limit,
        force=force,
        retry_errors=retry_errors,
        stage_name=f"verify ({config.llm.model})",
    )
