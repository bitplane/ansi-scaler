from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm

from ansi_scaler.dataset.compiler import compile_dataset
from ansi_scaler.dataset.models import load_dataset_recipe
from ansi_scaler.dataset.reader import CompiledDataset
from ansi_scaler.identity import stable_id
from ansi_scaler.refiner.config import RefinerConfig
from ansi_scaler.refiner.model import LocalAnsiRefiner, RefinerOutput, refiner_loss
from ansi_scaler.refiner.sampler import Patch, PatchSampler, fixed_patches, load_patch_assets
from ansi_scaler.refiner.tracking import RefinerTracker


def _device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


def _batch(patches: list[Patch], device: torch.device) -> dict[str, torch.Tensor]:
    def array(name: str) -> np.ndarray:
        return np.stack([getattr(patch, name) for patch in patches])

    metadata = np.concatenate(
        [array("bbox"), array("scale"), array("source_lod_weights"), array("target_lod_weights")], axis=1
    )
    return {
        "glyphs": torch.from_numpy(array("context_glyphs").reshape((-1, 32))).long().to(device),
        "foreground": torch.from_numpy(array("context_foreground").reshape((-1, 32, 3))).float().to(device) / 255,
        "background": torch.from_numpy(array("context_background").reshape((-1, 32, 3))).float().to(device) / 255,
        "background_present": torch.from_numpy(array("context_background_present").reshape((-1, 32))).float().to(device),
        "metadata": torch.from_numpy(metadata).float().to(device),
        "target_glyphs": torch.from_numpy(array("target_glyphs").reshape((-1, 18))).long().to(device),
        "target_foreground": torch.from_numpy(array("target_foreground").reshape((-1, 18, 3))).float().to(device) / 255,
        "target_background": torch.from_numpy(array("target_background").reshape((-1, 18, 3))).float().to(device) / 255,
        "target_background_present": torch.from_numpy(array("target_background_present").reshape((-1, 18))).float().to(device),
    }


def _forward(model: LocalAnsiRefiner, batch: dict[str, torch.Tensor]) -> RefinerOutput:
    return model(*(batch[name] for name in ("glyphs", "foreground", "background", "background_present", "metadata")))


def _nearest(patches: list[Patch]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_x = np.floor((np.arange(6) + 0.5) * 4 / 6).astype(int)
    source_y = np.floor((np.arange(3) + 0.5) * 2 / 3).astype(int)
    core = (slice(1, 3), slice(2, 6))
    outputs = []
    for name in ("context_glyphs", "context_foreground", "context_background", "context_background_present"):
        values = np.stack([getattr(patch, name)[core] for patch in patches])
        outputs.append(values[:, source_y[:, None], source_x[None, :]])
    return tuple(outputs)  # type: ignore[return-value]


def _font() -> ImageFont.FreeTypeFont:
    path = Path(__file__).parents[1] / "review" / "static" / "fonts" / "NotoSansSymbols2-Regular.ttf"
    return ImageFont.truetype(path, 16)


def _render(glyphs: np.ndarray, foreground: np.ndarray, background: np.ndarray, present: np.ndarray, codepoints: list[int]) -> Image.Image:
    image = Image.new("RGB", (48, 48), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = _font()
    for y in range(3):
        for x in range(6):
            if present[y, x]:
                draw.rectangle((x * 8, y * 16, x * 8 + 7, y * 16 + 15), fill=tuple(int(v) for v in background[y, x]))
            glyph_id = int(glyphs[y, x])
            codepoint = codepoints[glyph_id - 3] if glyph_id >= 3 and glyph_id - 3 < len(codepoints) else ord("?")
            if codepoint != ord(" "):
                draw.text((x * 8 + 4, y * 16 + 8), chr(codepoint), font=font, fill=tuple(int(v) for v in foreground[y, x]), anchor="mm")
    return image


def evaluate(
    model: LocalAnsiRefiner, patches: list[Patch], dataset: CompiledDataset,
    config: RefinerConfig, device: torch.device, *, contact_path: Path | None = None,
) -> dict[str, float]:
    if not patches:
        return {}
    model.eval()
    totals: dict[str, float] = {"glyph_accuracy": 0, "presence_accuracy": 0, "rgb_mae": 0, "render_mse": 0, "baseline_render_mse": 0}
    samples = 0
    contact_rows = []
    codepoints = dataset.vocabulary["codepoints"]
    space_id = codepoints.index(ord(" ")) + 3
    with torch.inference_mode():
        for start in range(0, len(patches), config.batch_size):
            chunk = patches[start : start + config.batch_size]
            batch = _batch(chunk, device)
            output = _forward(model, batch)
            glyphs = output.glyph_logits.argmax(-1).cpu().numpy().reshape((-1, 3, 6))
            foreground = (output.foreground.cpu().numpy().reshape((-1, 3, 6, 3)) * 255).round().astype(np.uint8)
            background = (output.background.cpu().numpy().reshape((-1, 3, 6, 3)) * 255).round().astype(np.uint8)
            present = (output.background_logits.sigmoid().cpu().numpy().reshape((-1, 3, 6)) >= 0.5)
            targets = batch["target_glyphs"].cpu().numpy().reshape((-1, 3, 6))
            target_fg = (batch["target_foreground"].cpu().numpy().reshape((-1, 3, 6, 3)) * 255).round().astype(np.uint8)
            target_bg = (batch["target_background"].cpu().numpy().reshape((-1, 3, 6, 3)) * 255).round().astype(np.uint8)
            target_present = batch["target_background_present"].bool().cpu().numpy().reshape((-1, 3, 6))
            near = _nearest(chunk)
            for index in range(len(chunk)):
                totals["glyph_accuracy"] += float((glyphs[index] == targets[index]).mean())
                totals["presence_accuracy"] += float((present[index] == target_present[index]).mean())
                mask = targets[index] != space_id
                totals["rgb_mae"] += float(np.abs(foreground[index].astype(float) - target_fg[index]).mean(axis=-1)[mask].mean()) if mask.any() else 0
                predicted_image = _render(glyphs[index], foreground[index], background[index], present[index], codepoints)
                target_image = _render(targets[index], target_fg[index], target_bg[index], target_present[index], codepoints)
                baseline_image = _render(near[0][index], near[1][index], near[2][index], near[3][index], codepoints)
                rendered_errors, baseline_errors = [], []
                for size in ((48, 48), (12, 12), (6, 3)):
                    target_array = np.asarray(target_image.resize(size, Image.Resampling.BICUBIC), dtype=np.float32) / 255
                    rendered_errors.append(float(np.square(np.asarray(predicted_image.resize(size, Image.Resampling.BICUBIC), dtype=np.float32) / 255 - target_array).mean()))
                    baseline_errors.append(float(np.square(np.asarray(baseline_image.resize(size, Image.Resampling.BICUBIC), dtype=np.float32) / 255 - target_array).mean()))
                totals["render_mse"] += sum(rendered_errors) / len(rendered_errors)
                totals["baseline_render_mse"] += sum(baseline_errors) / len(baseline_errors)
                if len(contact_rows) < 32:
                    row = Image.new("RGB", (48 * 3, 48))
                    for column, value in enumerate((baseline_image, predicted_image, target_image)):
                        row.paste(value, (column * 48, 0))
                    contact_rows.append(row.resize((48 * 3 * 4, 48 * 4), Image.Resampling.NEAREST))
                samples += 1
    if contact_path and contact_rows:
        sheet = Image.new("RGB", (contact_rows[0].width, sum(row.height for row in contact_rows)), "black")
        for index, row in enumerate(contact_rows):
            sheet.paste(row, (0, index * row.height))
        contact_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(contact_path)
    return {key: value / samples for key, value in totals.items()}


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _train_refiner(config: RefinerConfig) -> Path:
    recipe = load_dataset_recipe(config.dataset_recipe)
    dataset_path = compile_dataset(recipe)
    dataset = CompiledDataset.open(dataset_path)
    vocabulary_size = len(dataset.vocabulary["codepoints"]) + 3
    space_id = dataset.vocabulary["codepoints"].index(ord(" ")) + 3
    train_assets = load_patch_assets(dataset, "train")
    if config.train_asset_limit:
        train_assets = train_assets[: config.train_asset_limit]
    validation_assets = load_patch_assets(dataset, "validation")
    test_assets = load_patch_assets(dataset, "test")
    sampler = PatchSampler(train_assets, config.seed)
    validation = fixed_patches(validation_assets, config.eval_patches_per_asset, config.seed + 1)
    test = fixed_patches(test_assets, config.eval_patches_per_asset, config.seed + 2)
    run_id = stable_id("local-ansi-refiner-v2", config.model_dump(mode="json"), dataset.metadata["dataset_id"])
    run_dir = config.output_root / config.name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.model_dump(mode="json"), sort_keys=True, indent=2) + "\n")
    device = _device(config.device)
    model = LocalAnsiRefiner(vocabulary_size, config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    tracker = RefinerTracker.start(
        output_root=config.output_root,
        experiment=config.mlflow_experiment,
        run_dir=run_dir,
        run_name=f"{config.name}-{run_id[:12]}",
        log_steps=config.mlflow_log_steps,
        parameters={
            "config": config.model_dump(mode="json"),
            "dataset": {
                "id": dataset.metadata["dataset_id"],
                "vocabulary_sha256": dataset.metadata["vocabulary_sha256"],
                "path": str(dataset_path),
                "train_assets": len(train_assets),
                "validation_assets": len(validation_assets),
                "test_assets": len(test_assets),
            },
            "model": {"parameters": parameter_count, "vocabulary_size": vocabulary_size},
            "runtime": {"device": str(device), "torch": torch.__version__},
        },
        tags={
            "ansi_scaler.run_id": run_id,
            "ansi_scaler.dataset_id": dataset.metadata["dataset_id"],
            "ansi_scaler.model": "local-ansi-refiner-v2",
        },
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / max(1, config.warmup_steps))
        * 0.5 * (1 + math.cos(math.pi * max(0, step - config.warmup_steps) / max(1, config.steps - config.warmup_steps))),
    )
    checkpoint = run_dir / "last.pt"
    start = 0
    best = math.inf
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state["dataset_id"] != dataset.metadata["dataset_id"] or state["vocabulary_sha256"] != dataset.metadata["vocabulary_sha256"]:
            raise ValueError("Checkpoint dataset or glyph vocabulary does not match this run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, best = state["step"], state["best"]
    metrics_path = run_dir / "metrics.jsonl"
    progress = tqdm(range(start, config.steps), desc="train refiner", unit="step", dynamic_ncols=True)
    for step in progress:
        model.train()
        batch = _batch([sampler.sample() for _ in range(config.batch_size)], device)
        optimizer.zero_grad(set_to_none=True)
        amp = device.type == "cuda"
        dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            output = _forward(model, batch)
            loss, parts = refiner_loss(
                output, batch["target_glyphs"], batch["target_foreground"], batch["target_background"],
                batch["target_background_present"], space_id, config,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        progress.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.2g}")
        record = {"step": step + 1, "split": "train", "lr": scheduler.get_last_lr()[0], **{key: value.item() for key, value in parts.items()}}
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        tracker.metrics({key: value for key, value in record.items() if key not in ("step", "split")}, step=step + 1, prefix="train")
        should_eval = (step + 1) % config.eval_steps == 0 or step + 1 == config.steps
        if should_eval and validation:
            values = evaluate(model, validation, dataset, config, device)
            with metrics_path.open("a") as handle:
                handle.write(json.dumps({"step": step + 1, "split": "validation", **values}, sort_keys=True) + "\n")
            tracker.metrics(values, step=step + 1, prefix="validation", force=True)
            if values["render_mse"] < best:
                best = values["render_mse"]
                save_file(model.state_dict(), run_dir / "best.safetensors")
        if (step + 1) % config.checkpoint_steps == 0 or step + 1 == config.steps:
            _atomic_checkpoint(checkpoint, {
                "step": step + 1, "best": best, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "dataset_id": dataset.metadata["dataset_id"],
                "vocabulary_sha256": dataset.metadata["vocabulary_sha256"],
            })
    best_path = run_dir / "best.safetensors"
    if best_path.exists():
        model.load_state_dict(load_file(best_path, device=str(device)))
    test_values = evaluate(model, test, dataset, config, device, contact_path=run_dir / "test-contact-sheet.png")
    report = {
        "run_id": run_id, "dataset": str(dataset_path), "assets": {"train": len(train_assets), "validation": len(validation_assets), "test": len(test_assets)},
        "test": test_values, "beats_nearest": bool(test_values and test_values["render_mse"] < test_values["baseline_render_mse"]),
    }
    (run_dir / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    tracker.metrics(test_values, step=config.steps, prefix="test", force=True)
    for artifact in (run_dir / "config.json", run_dir / "report.json", run_dir / "test-contact-sheet.png", best_path):
        tracker.artifact(artifact)
    tracker.finish()
    return run_dir


def train_refiner(config: RefinerConfig) -> Path:
    try:
        return _train_refiner(config)
    except BaseException:
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.end_run(status="FAILED")
        raise
