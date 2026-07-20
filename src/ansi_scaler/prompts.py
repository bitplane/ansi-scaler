from __future__ import annotations

from pathlib import Path
from typing import Any

from ansi_scaler.config import RunConfig
from ansi_scaler.content import ContentLibrary
from ansi_scaler.identity import stable_id
from ansi_scaler.manifests import write_jsonl


def build_prompt_records(config: RunConfig, content: ContentLibrary) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    request_index = 0
    for location, specification in content.objects:
        specification_id = f"{location.theme}/{location.location}/{specification.id}"
        prompt = f"{specification.prompt.rstrip(' ,')}, {config.sana.presentation_prompt}"
        exclusions = list(dict.fromkeys([*specification.exclusions, *config.sana.exclusions]))
        negative_prompt = ", ".join(exclusions)
        prompt_family_id = stable_id(
            "authored-object-v1", specification_id, specification.model_dump(mode="json"), prompt, negative_prompt
        )
        for seed_index in range(config.prompts.seeds_per_specification):
            seed = config.prompts.base_seed + request_index
            request_index += 1
            records.append(
                {
                    "id": stable_id("generation-request-v2", prompt_family_id, seed_index, seed),
                    "stage": "prompts",
                    "contract": "authored-object-prompts-v1",
                    "prompt_family_id": prompt_family_id,
                    "specification_id": specification_id,
                    "object_id": specification.id,
                    "label": specification.label,
                    "subject_family": specification.subject_family,
                    "theme": location.theme,
                    "location": location.location,
                    "tags": specification.tags,
                    "semantic_prompt": specification.prompt,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "seed_index": seed_index,
                    "seed": seed,
                }
            )
    return records


def write_prompt_manifest(config: RunConfig, content: ContentLibrary) -> Path:
    destination = config.manifest_dir / "prompts.jsonl"
    records = build_prompt_records(config, content)
    if config.limit is not None:
        records = records[: config.limit]
    write_jsonl(destination, records)
    return destination
