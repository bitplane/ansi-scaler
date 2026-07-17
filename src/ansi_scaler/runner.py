from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

from ansi_scaler.manifests import append_jsonl, known_ids, read_jsonl


Processor = Callable[[dict[str, Any]], dict[str, Any]]
IdBuilder = Callable[[dict[str, Any]], str]


def run_stage(
    inputs: Iterable[dict[str, Any]],
    output_manifest: Path,
    error_manifest: Path,
    processor: Processor,
    id_builder: IdBuilder,
    *,
    limit: int | None = None,
    force: bool = False,
    retry_errors: bool = False,
) -> tuple[int, int, int]:
    if force:
        output_manifest.unlink(missing_ok=True)
        error_manifest.unlink(missing_ok=True)

    completed = known_ids(output_manifest)
    failed_parents = {record["parent_id"] for record in read_jsonl(error_manifest)} if not retry_errors else set()
    successes = failures = skipped = considered = 0

    for source in inputs:
        if limit is not None and considered >= limit:
            break
        considered += 1
        parent_id = source["id"]
        output_id = id_builder(source)
        if output_id in completed:
            skipped += 1
            continue
        if parent_id in failed_parents:
            skipped += 1
            continue
        try:
            result = processor(source)
            if result["id"] != output_id:
                raise ValueError(f"Processor returned unexpected id {result['id']}; expected {output_id}")
            append_jsonl(output_manifest, result)
            completed.add(result["id"])
            successes += 1
        except Exception as error:  # noqa: BLE001 - batch processing must preserve later records
            append_jsonl(
                error_manifest,
                {
                    "parent_id": parent_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            failures += 1
    return successes, failures, skipped
