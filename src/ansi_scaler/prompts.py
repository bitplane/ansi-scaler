from __future__ import annotations

from pathlib import Path
from typing import Any

from ansi_scaler.catalog import Catalog
from ansi_scaler.config import RunConfig, load_yaml
from ansi_scaler.identity import stable_id
from ansi_scaler.manifests import write_jsonl


def _format_subject(name: str, variant: str | None) -> str:
    return variant or name


def build_prompt_records(config: RunConfig, catalog: Catalog) -> list[dict[str, Any]]:
    template_config = load_yaml(config.prompts.template_config)
    templates = template_config["templates"]
    style = template_config["style"]
    negative_prompt = template_config["negative_prompt"]
    if len(templates) < config.prompts.variants_per_entry:
        raise ValueError("Prompt configuration does not contain enough templates")

    records: list[dict[str, Any]] = []
    request_index = 0
    for kit_id in sorted(catalog.kits):
        kit = catalog.kits[kit_id]
        for role, entry in kit.entries():
            concept = catalog.concepts[entry.concept]
            subject = _format_subject(concept.name, entry.variant)
            membership = {
                "kit_id": kit.id,
                "kit_name": kit.name,
                "concept_id": concept.id,
                "concept_name": concept.name,
                "role": role,
                "variant": entry.variant,
                "attributes": entry.attributes,
                "theme": kit.theme,
            }
            for variant_index in range(config.prompts.variants_per_entry):
                prompt = templates[variant_index].format(subject=subject, style=style)
                prompt_id = stable_id("prompt-v1", membership, variant_index, prompt, negative_prompt)
                for seed_index in range(config.prompts.seeds_per_prompt):
                    seed = config.prompts.base_seed + request_index
                    request_index += 1
                    record_id = stable_id("generation-request-v1", prompt_id, seed_index, seed)
                    records.append(
                        {
                            "id": record_id,
                            "stage": "prompts",
                            "prompt_id": prompt_id,
                            "variant_index": variant_index,
                            "seed_index": seed_index,
                            "seed": seed,
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            **membership,
                        }
                    )
    return records


def write_prompt_manifest(config: RunConfig, catalog: Catalog) -> Path:
    destination = config.manifest_dir / "prompts.jsonl"
    records = build_prompt_records(config, catalog)
    if config.limit is not None:
        records = records[: config.limit]
    write_jsonl(destination, records)
    return destination
