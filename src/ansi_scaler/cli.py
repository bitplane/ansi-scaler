from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from ansi_scaler.catalog import load_catalog
from ansi_scaler.config import RunConfig, load_run_config
from ansi_scaler.prompts import write_prompt_manifest
from ansi_scaler.stages.classify import run_classify
from ansi_scaler.stages.generate import run_generate
from ansi_scaler.stages.lod import run_lod
from ansi_scaler.stages.rembg import run_rembg
from ansi_scaler.stages.verify import run_verify


app = typer.Typer(help="Build reproducible synthetic ANSI-art training corpora.", no_args_is_help=True)
catalog_app = typer.Typer(help="Inspect and validate the source catalogue.")
prompts_app = typer.Typer(help="Build deterministic generation prompts.")
app.add_typer(catalog_app, name="catalog")
app.add_typer(prompts_app, name="prompts")

RunConfigOption = Annotated[Path, typer.Option("--run-config", exists=True, dir_okay=False)]


def _config(path: Path) -> RunConfig:
    config = load_run_config(path)
    if ollama_host := os.environ.get("OLLAMA_HOST"):
        config.vlm.endpoint = ollama_host
        config.llm.endpoint = ollama_host
    return config


def _result(stage: str, result: tuple[int, int, int]) -> None:
    successes, failures, skipped = result
    typer.echo(f"{stage}: {successes} completed, {failures} failed, {skipped} skipped")
    if failures:
        raise typer.Exit(code=1)


@catalog_app.command("validate")
def catalog_validate(run_config: RunConfigOption) -> None:
    """Validate concepts, roles, references, and kit budgets."""
    config = _config(run_config)
    catalog = load_catalog(config.catalog_dir)
    typer.echo(
        f"Valid catalogue: {len(catalog.concepts)} concepts, {len(catalog.kits)} kits, "
        f"{catalog.membership_count} memberships"
    )


@prompts_app.command("build")
def prompts_build(run_config: RunConfigOption) -> None:
    """Build the deterministic JSONL generation-request manifest."""
    config = _config(run_config)
    catalog = load_catalog(config.catalog_dir)
    destination = write_prompt_manifest(config, catalog)
    typer.echo(f"Wrote {destination}")


@app.command("generate")
def generate(
    run_config: RunConfigOption,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    force: bool = False,
    retry_errors: bool = False,
) -> None:
    """Generate raster objects with the configured Sana model."""
    _result("generate", run_generate(_config(run_config), limit=limit, force=force, retry_errors=retry_errors))


@app.command("rembg")
def remove_background(
    run_config: RunConfigOption,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    force: bool = False,
    retry_errors: bool = False,
) -> None:
    """Remove generated backgrounds with the configured rembg model."""
    _result("rembg", run_rembg(_config(run_config), limit=limit, force=force, retry_errors=retry_errors))


@app.command("lod")
def lod(
    run_config: RunConfigOption,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    force: bool = False,
    retry_errors: bool = False,
) -> None:
    """Generate canonical SVG LODs and small PNG previews."""
    _result("lod", run_lod(_config(run_config), limit=limit, force=force, retry_errors=retry_errors))


@app.command("classify")
def classify(
    run_config: RunConfigOption,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    artifact_ids: Annotated[list[str] | None, typer.Option("--artifact-id")] = None,
    force: bool = False,
    retry_errors: bool = False,
) -> None:
    """Describe cutouts with the configured local vision-language model."""
    _result(
        "classify",
        run_classify(
            _config(run_config),
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
    _result(
        "verify",
        run_verify(
            _config(run_config),
            limit=limit,
            artifact_ids=set(artifact_ids or []),
            force=force,
            retry_errors=retry_errors,
        ),
    )


@app.command("run")
def run_pipeline(
    run_config: RunConfigOption,
    through: Annotated[str, typer.Option()] = "lod",
    force: bool = False,
    retry_errors: bool = False,
) -> None:
    """Run or resume the corpus pipeline through a named stage."""
    stages = ["prompts", "generate", "rembg", "lod"]
    if through not in stages:
        raise typer.BadParameter(f"Expected one of: {', '.join(stages)}")
    config = _config(run_config)
    catalog = load_catalog(config.catalog_dir)
    write_prompt_manifest(config, catalog)
    if through == "prompts":
        return
    _result("generate", run_generate(config, force=force, retry_errors=retry_errors))
    if through == "generate":
        return
    _result("rembg", run_rembg(config, force=force, retry_errors=retry_errors))
    if through == "rembg":
        return
    _result("lod", run_lod(config, force=force, retry_errors=retry_errors))
