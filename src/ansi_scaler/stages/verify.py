from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ansi_scaler.config import RunConfig
from ansi_scaler.identity import stable_id
from ansi_scaler.manifests import read_jsonl
from ansi_scaler.runner import run_stage
from ansi_scaler.stages.ollama import RequestFunction, request_with_retry


VERIFIER_PROMPT = """You are a conservative quality gate for a synthetic game-art corpus.
Compare the requested concept and prompt with an independent vision model's factual observations.
Reject when the visible primary object is a different kind of object, the result is incoherent, the prompt requests one
isolated asset but several candidate assets are present, or reported artefacts make it unsuitable. Minor stylistic,
colour, viewpoint, or decorative differences are acceptable. If evidence is ambiguous or insufficient, choose review,
never accept. Return only the requested structured result."""


class Verification(BaseModel):
    semantic_match: bool
    cardinality_match: bool
    visually_usable: bool
    decision: Literal["accept", "reject", "review"]
    rejection_reasons: list[str]
    explanation: str
    uncertainty: float = Field(ge=0.0, le=1.0)


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
            "generation_prompt": source["prompt"],
            "vision_observations": source["classification"],
        }
        payload = {
            "model": self.settings.model,
            "stream": False,
            "think": False,
            "format": Verification.model_json_schema(),
            "keep_alive": self.settings.keep_alive,
            "options": {"temperature": self.settings.temperature},
            "messages": [
                {"role": "system", "content": VERIFIER_PROMPT},
                {"role": "user", "content": json.dumps(evidence, sort_keys=True)},
            ],
        }
        response = request_with_retry(
            self.request_function, payload, self.settings, service=f"LLM server {self.settings.endpoint}"
        )
        self.used_model = True
        verification = Verification.model_validate_json(response["message"]["content"])
        observations = source["classification"]
        if observations["multiple_candidate_assets"] or observations["object_count"] != 1:
            verification.cardinality_match = False
            verification.decision = "reject"
            reason = f"Vision classifier found {observations['object_count']} separate candidate assets; expected one."
            if reason not in verification.rejection_reasons:
                verification.rejection_reasons.append(reason)
        if verification.decision == "accept" and (
            not verification.semantic_match
            or not verification.cardinality_match
            or not verification.visually_usable
            or verification.uncertainty >= 0.5
        ):
            raise ValueError("Verifier returned an internally inconsistent acceptance")
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
    inputs = read_jsonl(config.manifest_dir / "classifications.jsonl")
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
