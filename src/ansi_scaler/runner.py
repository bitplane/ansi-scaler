from __future__ import annotations

import traceback
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm.auto import tqdm

from ansi_scaler.manifests import append_jsonl, known_ids, read_jsonl


Processor = Callable[[dict[str, Any]], dict[str, Any]]
IdBuilder = Callable[[dict[str, Any]], str]


def _error_record(parent_id: str, error: Exception) -> dict[str, str]:
    return {
        "parent_id": parent_id,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }


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
    stage_name: str = "stage",
    show_progress: bool = True,
) -> tuple[int, int, int]:
    if force:
        output_manifest.unlink(missing_ok=True)
        error_manifest.unlink(missing_ok=True)

    completed = known_ids(output_manifest)
    failed_parents = {record["parent_id"] for record in read_jsonl(error_manifest)} if not retry_errors else set()
    selected_inputs = list(islice(inputs, limit)) if limit is not None else list(inputs)
    pending = []
    scheduled_ids = set(completed)
    skipped = 0
    for source in selected_inputs:
        output_id = id_builder(source)
        if output_id in scheduled_ids or source["id"] in failed_parents:
            skipped += 1
            continue
        pending.append(source)
        scheduled_ids.add(output_id)
    successes = failures = 0

    progress = tqdm(
        pending,
        desc=stage_name,
        unit="image",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    progress.set_postfix(completed=0, failed=0, skipped=skipped, refresh=False)
    try:
        for source in progress:
            parent_id = source["id"]
            output_id = id_builder(source)
            try:
                result = processor(source)
                if result["id"] != output_id:
                    raise ValueError(f"Processor returned unexpected id {result['id']}; expected {output_id}")
                append_jsonl(output_manifest, result)
                completed.add(result["id"])
                successes += 1
            except Exception as error:  # noqa: BLE001 - batch processing must preserve later records
                append_jsonl(error_manifest, _error_record(parent_id, error))
                failures += 1
            progress.set_postfix(completed=successes, failed=failures, skipped=skipped, refresh=False)
    finally:
        progress.close()
        close = getattr(processor, "close", None)
        if callable(close):
            close()
    return successes, failures, skipped


def run_parallel_stage(
    inputs: Iterable[dict[str, Any]],
    output_manifest: Path,
    error_manifest: Path,
    processor: Processor,
    id_builder: IdBuilder,
    *,
    workers: int,
    limit: int | None = None,
    force: bool = False,
    retry_errors: bool = False,
    stage_name: str = "stage",
) -> tuple[int, int, int]:
    if workers <= 1:
        return run_stage(
            inputs,
            output_manifest,
            error_manifest,
            processor,
            id_builder,
            limit=limit,
            force=force,
            retry_errors=retry_errors,
            stage_name=stage_name,
        )
    if force:
        output_manifest.unlink(missing_ok=True)
        error_manifest.unlink(missing_ok=True)

    completed = known_ids(output_manifest)
    failed_parents = {record["parent_id"] for record in read_jsonl(error_manifest)} if not retry_errors else set()
    selected = list(islice(inputs, limit)) if limit is not None else list(inputs)
    pending: list[tuple[dict[str, Any], str]] = []
    scheduled_ids = set(completed)
    skipped = 0
    for source in selected:
        output_id = id_builder(source)
        if output_id in scheduled_ids or source["id"] in failed_parents:
            skipped += 1
        else:
            pending.append((source, output_id))
            scheduled_ids.add(output_id)

    workers = min(workers, len(pending))
    if workers <= 1:
        return run_stage(
            selected,
            output_manifest,
            error_manifest,
            processor,
            id_builder,
            force=False,
            retry_errors=retry_errors,
            stage_name=stage_name,
        )

    successes = failures = 0
    progress = tqdm(total=len(pending), desc=stage_name, unit="image", dynamic_ncols=True)
    progress.set_postfix(completed=0, failed=0, skipped=skipped, refresh=False)
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
            futures = {executor.submit(processor, source): (source, output_id) for source, output_id in pending}
            for future in as_completed(futures):
                source, output_id = futures[future]
                try:
                    result = future.result()
                    if result["id"] != output_id:
                        raise ValueError(f"Processor returned unexpected id {result['id']}; expected {output_id}")
                    append_jsonl(output_manifest, result)
                    completed.add(result["id"])
                    successes += 1
                except Exception as error:  # noqa: BLE001 - batch processing must preserve later records
                    append_jsonl(error_manifest, _error_record(source["id"], error))
                    failures += 1
                progress.update()
                progress.set_postfix(completed=successes, failed=failures, skipped=skipped, refresh=False)
    finally:
        progress.close()
    return successes, failures, skipped
