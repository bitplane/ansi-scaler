from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import typer

from ansi_scaler.config import RunConfig, load_run_config
from ansi_scaler.content import load_content
from ansi_scaler.dataset.compiler import compile_dataset, plan_dataset
from ansi_scaler.dataset.models import load_dataset_recipe
from ansi_scaler.dataset.validate import validate_dataset
from ansi_scaler.gc import apply_gc_plan, build_gc_plan, plan_report
from ansi_scaler.locking import CorpusBusyError, corpus_lock
from ansi_scaler.prompts import write_prompt_manifest
from ansi_scaler.stages.classify import run_classify
from ansi_scaler.stages.generate import run_generate
from ansi_scaler.stages.lod import run_lod
from ansi_scaler.stages.pyramid import run_pyramid
from ansi_scaler.stages.rembg import run_rembg
from ansi_scaler.stages.verify import run_verify


app = typer.Typer(help="Build reproducible synthetic ANSI-art training corpora.", no_args_is_help=True)
content_app = typer.Typer(help="Inspect and validate authored object specifications.")
prompts_app = typer.Typer(help="Build deterministic generation prompts.")
app.add_typer(content_app, name="content")
app.add_typer(prompts_app, name="prompts")

RunConfigOption = Annotated[Path, typer.Option("--run-config", exists=True, dir_okay=False)]
DatasetRecipeOption = Annotated[Path, typer.Option("--recipe", exists=True, dir_okay=False)]


def _config(path: Path) -> RunConfig:
    config = load_run_config(path)
    if ollama_host := os.environ.get("OLLAMA_HOST"):
        config.vlm.endpoint = ollama_host
        config.llm.endpoint = ollama_host
    return config


def _result(stage: str, result: tuple[int, int, int], *, fail_on_record_errors: bool = True) -> None:
    successes, failures, skipped = result
    typer.echo(f"{stage}: {successes} completed, {failures} failed, {skipped} skipped")
    if failures and fail_on_record_errors:
        raise typer.Exit(code=1)


@content_app.command("validate")
def content_validate(run_config: RunConfigOption) -> None:
    """Validate the single-path theme/location/object library."""
    config = _config(run_config)
    content = load_content(config.content_dir)
    themes = {location.theme for location in content.locations}
    typer.echo(
        f"Valid content: {len(content.objects)} objects, {len(content.locations)} locations, {len(themes)} themes"
    )


@prompts_app.command("build")
def prompts_build(run_config: RunConfigOption) -> None:
    """Build the deterministic JSONL generation-request manifest."""
    config = _config(run_config)
    content = load_content(config.content_dir)
    with corpus_lock(config.data_dir, exclusive=False):
        destination = write_prompt_manifest(config, content)
    typer.echo(f"Wrote {destination}")


@app.command("generate")
def generate(
    run_config: RunConfigOption,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    force: bool = False,
    retry_errors: bool = False,
) -> None:
    """Generate raster objects with the configured Sana model."""
    config = _config(run_config)
    with corpus_lock(config.data_dir, exclusive=False):
        _result("generate", run_generate(config, limit=limit, force=force, retry_errors=retry_errors))


@app.command("rembg")
def remove_background(
    run_config: RunConfigOption,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    force: bool = False,
    retry_errors: bool = False,
) -> None:
    """Remove generated backgrounds with the configured rembg model."""
    config = _config(run_config)
    with corpus_lock(config.data_dir, exclusive=False):
        _result("rembg", run_rembg(config, limit=limit, force=force, retry_errors=retry_errors))


@app.command("lod")
def lod(
    run_config: RunConfigOption,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    force: bool = False,
    retry_errors: bool = False,
) -> None:
    """Generate canonical SVG LODs and small PNG previews."""
    config = _config(run_config)
    with corpus_lock(config.data_dir, exclusive=False):
        _result("lod", run_lod(config, limit=limit, force=force, retry_errors=retry_errors))


@app.command("pyramid")
def pyramid(
    run_config: RunConfigOption,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    force: bool = False,
    retry_errors: bool = False,
) -> None:
    """Build compressed multi-scale ANSI pyramids with Chuda."""
    config = _config(run_config)
    with corpus_lock(config.data_dir, exclusive=False):
        _result("pyramid", run_pyramid(config, limit=limit, force=force, retry_errors=retry_errors))


@app.command("classify")
def classify(
    run_config: RunConfigOption,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    artifact_ids: Annotated[list[str] | None, typer.Option("--artifact-id")] = None,
    force: bool = False,
    retry_errors: bool = False,
) -> None:
    """Describe cutouts with the configured local vision-language model."""
    config = _config(run_config)
    with corpus_lock(config.data_dir, exclusive=False):
        _result(
            "classify",
            run_classify(
                config,
                limit=limit,
                artifact_ids=set(artifact_ids or []),
                force=force,
                retry_errors=retry_errors,
            ),
        )


@app.command("verify")
def verify(
    run_config: RunConfigOption,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    artifact_ids: Annotated[list[str] | None, typer.Option("--artifact-id")] = None,
    force: bool = False,
    retry_errors: bool = False,
) -> None:
    """Verify VLM observations against prompts with a text-only LLM."""
    config = _config(run_config)
    with corpus_lock(config.data_dir, exclusive=False):
        _result(
            "verify",
            run_verify(
                config,
                limit=limit,
                artifact_ids=set(artifact_ids or []),
                force=force,
                retry_errors=retry_errors,
            ),
        )


@app.command("review")
def review(
    run_config: RunConfigOption,
    host: str | None = None,
    port: Annotated[int | None, typer.Option(min=1, max=65535)] = None,
) -> None:
    """Serve the local corpus review interface."""
    import uvicorn

    from ansi_scaler.review.web import create_app

    config = _config(run_config)
    listen_host = host or config.review.host
    listen_port = port or config.review.port
    typer.echo(f"Review UI: http://{listen_host}:{listen_port}")
    with corpus_lock(config.data_dir, exclusive=False):
        uvicorn.run(create_app(config), host=listen_host, port=listen_port, log_level="info")


@app.command("run")
def run_pipeline(
    run_config: RunConfigOption,
    through: Annotated[str, typer.Option()] = "pyramid",
    force: bool = False,
    retry_errors: bool = False,
) -> None:
    """Run or resume the corpus pipeline through a named stage."""
    stages = ["prompts", "generate", "rembg", "lod", "pyramid", "classify", "verify"]
    if through not in stages:
        raise typer.BadParameter(f"Expected one of: {', '.join(stages)}")
    config = _config(run_config)
    with corpus_lock(config.data_dir, exclusive=False):
        content = load_content(config.content_dir)
        write_prompt_manifest(config, content)
        if through == "prompts":
            return
        _result("generate", run_generate(config, force=force, retry_errors=retry_errors), fail_on_record_errors=False)
        if through == "generate":
            return
        _result("rembg", run_rembg(config, force=force, retry_errors=retry_errors), fail_on_record_errors=False)
        if through == "rembg":
            return
        _result("lod", run_lod(config, force=force, retry_errors=retry_errors), fail_on_record_errors=False)
        if through == "lod":
            return
        _result("pyramid", run_pyramid(config, force=force, retry_errors=retry_errors), fail_on_record_errors=False)
        if through == "pyramid":
            return
        _result("classify", run_classify(config, force=force, retry_errors=retry_errors), fail_on_record_errors=False)
        if through == "classify":
            return
        _result("verify", run_verify(config, force=force, retry_errors=retry_errors), fail_on_record_errors=False)


@app.command("gc")
def garbage_collect(run_config: RunConfigOption, confirm: bool = False) -> None:
    """Safely compact superseded corpus records and unreferenced artifacts."""
    config = _config(run_config)
    try:
        with corpus_lock(config.data_dir, exclusive=True, blocking=False):
            plan = build_gc_plan(config, run_config.resolve())
            typer.echo(plan_report(plan))
            if plan.removed_records == 0 and not plan.delete_paths:
                return
            if not confirm:
                if not sys.stdin.isatty():
                    typer.echo("Non-interactive input: no changes made. Pass --confirm to apply this plan.")
                    return
                if not typer.confirm("Permanently delete these unreferenced artifacts?", default=False):
                    typer.echo("Cancelled; no changes made.")
                    return
            receipt = apply_gc_plan(plan)
            typer.echo(f"GC complete; backup and receipt: {receipt}")
    except (CorpusBusyError, ValueError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


@app.command("dataset-plan")
def dataset_plan(
    recipe: DatasetRecipeOption,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Report selection and estimated size without writing a dataset."""
    settings = load_dataset_recipe(recipe)
    report = plan_dataset(settings, limit=limit)
    typer.echo(json.dumps(report, sort_keys=True, indent=2))


@app.command("dataset")
def dataset_build(
    recipe: DatasetRecipeOption,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Compile selected pyramids into immutable safetensors shards."""
    settings = load_dataset_recipe(recipe)
    config = load_run_config(settings.run_config)
    with corpus_lock(config.data_dir, exclusive=False):
        typer.echo(f"Dataset: {compile_dataset(settings, limit=limit)}")


@app.command("dataset-validate")
def dataset_validate(path: Annotated[Path, typer.Option("--dataset-dir", exists=True, file_okay=False)]) -> None:
    """Validate checksums, tensors, offsets, splits, and dimensions."""
    typer.echo(json.dumps(validate_dataset(path), sort_keys=True, indent=2))
