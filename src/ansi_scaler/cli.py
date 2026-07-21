from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from ansi_scaler.config import RunConfig, load_run_config
from ansi_scaler.content import load_content
from ansi_scaler.dataset.compiler import compile_dataset, plan_dataset
from ansi_scaler.dataset.models import load_dataset_recipe
from ansi_scaler.dataset.reader import CompiledDataset
from ansi_scaler.dataset.validate import validate_dataset
from ansi_scaler.gc import apply_gc_plan, build_gc_plan, plan_report
from ansi_scaler.identity import stable_id
from ansi_scaler.locking import CorpusBusyError, corpus_lock
from ansi_scaler.prompts import write_prompt_manifest
from ansi_scaler.stages.classify import run_classify
from ansi_scaler.stages.generate import run_generate
from ansi_scaler.stages.lod import run_lod
from ansi_scaler.stages.pyramid import run_pyramid
from ansi_scaler.stages.background import run_background
from ansi_scaler.stages.verify import run_verify


app = typer.Typer(help="Build reproducible synthetic ANSI-art training corpora.", no_args_is_help=True)
content_app = typer.Typer(help="Inspect and validate authored object specifications.")
prompts_app = typer.Typer(help="Build deterministic generation prompts.")
app.add_typer(content_app, name="content")
app.add_typer(prompts_app, name="prompts")

RunConfigOption = Annotated[Path, typer.Option("--run-config", exists=True, dir_okay=False)]
DatasetRecipeOption = Annotated[Path, typer.Option("--recipe", exists=True, dir_okay=False)]
TrainingConfigOption = Annotated[Path, typer.Option("--training-config", exists=True, dir_okay=False)]


def _config(path: Path) -> RunConfig:
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    config = load_run_config(path)
    if ollama_host := os.environ.get("OLLAMA_HOST"):
        config.vlm.endpoint = ollama_host
        config.llm.endpoint = ollama_host
    return config


def _result(stage: str, result: tuple[int, int, int]) -> None:
    successes, failures, skipped = result
    typer.echo(f"{stage}: {successes} completed, {failures} failed, {skipped} skipped")
    if failures and not successes and not skipped:
        typer.echo(f"{stage}: every selected record failed; stopping the pipeline", err=True)
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


@app.command("background")
def background(
    run_config: RunConfigOption,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    force: bool = False,
    retry_errors: bool = False,
) -> None:
    """Extract RGBA subjects with the configured background provider."""
    config = _config(run_config)
    with corpus_lock(config.data_dir, exclusive=False):
        _result("background", run_background(config, limit=limit, force=force, retry_errors=retry_errors))


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
    stages = ["prompts", "generate", "background", "lod", "pyramid", "classify", "verify"]
    if through not in stages:
        raise typer.BadParameter(f"Expected one of: {', '.join(stages)}")
    config = _config(run_config)
    with corpus_lock(config.data_dir, exclusive=False):
        content = load_content(config.content_dir)
        write_prompt_manifest(config, content)
        if through == "prompts":
            return
        _result("generate", run_generate(config, force=force, retry_errors=retry_errors))
        if through == "generate":
            return
        _result("background", run_background(config, force=force, retry_errors=retry_errors))
        if through == "background":
            return
        _result("lod", run_lod(config, force=force, retry_errors=retry_errors))
        if through == "lod":
            return
        _result("pyramid", run_pyramid(config, force=force, retry_errors=retry_errors))
        if through == "pyramid":
            return
        _result("classify", run_classify(config, force=force, retry_errors=retry_errors))
        if through == "classify":
            return
        _result("verify", run_verify(config, force=force, retry_errors=retry_errors))


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


@app.command("refiner-train")
def refiner_train(training_config: TrainingConfigOption) -> None:
    """Train or resume the local 8x4-context ANSI refinement experiment."""
    from ansi_scaler.refiner.config import load_refiner_config
    from ansi_scaler.refiner.train import train_refiner

    settings = load_refiner_config(training_config)
    typer.echo(f"Refiner run: {train_refiner(settings)}")


@app.command("refiner-demo")
def refiner_demo(
    object_name: Annotated[str, typer.Argument(help="Object to generate")],
    run_config: RunConfigOption,
    training_config: TrainingConfigOption,
    width: Annotated[int, typer.Option(min=4)] = 40,
    checkpoint: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    seed: int = 42000,
) -> None:
    """Generate an object and print its Chuda and learned 1.5x ANSI forms."""
    import chuda
    import torch

    from ansi_scaler.ansi import decode_ansi, encode_ansi
    from ansi_scaler.refiner.config import load_refiner_config
    from ansi_scaler.refiner.inference import load_refiner, scale_cells
    from ansi_scaler.stages.generate import SanaGenerator

    config = _config(run_config)
    training = load_refiner_config(training_config)
    dataset = CompiledDataset.open(compile_dataset(load_dataset_recipe(training.dataset_recipe)))
    if checkpoint is None:
        candidates = list((training.output_root / training.name).glob("*/best.safetensors"))
        if not candidates:
            raise typer.BadParameter("No best.safetensors found; pass --checkpoint explicitly")
        checkpoint = max(candidates, key=lambda path: path.stat().st_mtime)

    prompt = f"{object_name.rstrip(' ,')}, {config.sana.presentation_prompt}"
    source = {
        "id": stable_id("refiner-demo", prompt, seed),
        "prompt": prompt,
        "negative_prompt": ", ".join(config.sana.exclusions),
        "seed": seed,
    }
    generator = SanaGenerator(config)
    try:
        generated = generator(source)
    finally:
        generator.close()
    image_path = config.data_dir / generated["artifact"]
    renderer = chuda.Renderer(config.chuda.backend, config.chuda.max_batch_cells)
    frame = renderer.render(
        chuda.Image.open(image_path), width, config.chuda.font_ratio, config.chuda.transparent_threshold
    )
    original_data = frame.to_ansi()
    original = decode_ansi(original_data, width=frame.columns, rows=frame.rows)
    device = torch.device(
        "cuda" if training.device == "auto" and torch.cuda.is_available()
        else "cpu" if training.device == "auto"
        else training.device
    )
    model = load_refiner(checkpoint, dataset, training, device)
    enlarged, enlarged_width, enlarged_rows = scale_cells(
        original,
        width=frame.columns,
        rows=frame.rows,
        model=model,
        dataset=dataset,
        config=training,
        lod_boundaries=(config.chuda.lod_3_below, config.chuda.lod_2_below, config.chuda.lod_1_below),
        device=device,
    )
    typer.echo(f"Generated raster: {image_path}")
    typer.echo(f"Checkpoint: {checkpoint}")
    typer.echo(f"\n--- Chuda {frame.columns}x{frame.rows} ---")
    sys.stdout.flush()
    sys.stdout.buffer.write(original_data)
    typer.echo(f"\n--- Refiner {enlarged_width}x{enlarged_rows} ---")
    sys.stdout.flush()
    sys.stdout.buffer.write(encode_ansi(enlarged, width=enlarged_width, rows=enlarged_rows))
