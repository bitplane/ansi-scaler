from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from ansi_scaler.config import load_run_config
from ansi_scaler.dataset.reader import CompiledDataset
from ansi_scaler.identity import stable_id
from ansi_scaler.refiner.config import RefinerConfig


FEATURE_FORMAT = "sana-prompt-features-v1"
SANA_INSTRUCTION = [
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
]


class PromptFeatures:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.index = json.loads((path / "index.json").read_text())
        tensors = load_file(path / "features.safetensors", device="cpu")
        self.embeddings = tensors["embeddings"]
        self.masks = tensors["masks"]
        self.by_asset = {asset_id: index for index, asset_id in enumerate(self.index["asset_ids"])}

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[-1])

    def get(self, asset_ids: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        indexes = torch.tensor([self.by_asset[value] for value in asset_ids], dtype=torch.long)
        return self.embeddings[indexes].to(device), self.masks[indexes].to(device)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _fake_features(prompts: list[str], length: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for prompt in prompts:
        seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:16], 16)
        generator = torch.Generator().manual_seed(seed)
        rows.append(torch.randn((length, 64), generator=generator, dtype=torch.float16))
    return torch.stack(rows), torch.ones((len(rows), length), dtype=torch.bool)


def _sana_features(
    prompts: list[str], model_id: str, revision: str, length: int, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    from transformers import AutoModel, AutoTokenizer

    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device.type == "cuda" else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", revision=revision)
    encoder = AutoModel.from_pretrained(
        model_id, subfolder="text_encoder", revision=revision, torch_dtype=dtype
    ).to(device).eval()
    prefix = "\n".join(SANA_INSTRUCTION)
    prefix_tokens = len(tokenizer.encode(prefix))
    maximum = prefix_tokens + length - 2
    embeddings, masks = [], []
    try:
        with torch.inference_mode():
            for start in range(0, len(prompts), batch_size):
                values = [prefix + prompt.lower().strip() for prompt in prompts[start : start + batch_size]]
                encoded = tokenizer(
                    values, padding="max_length", max_length=maximum, truncation=True,
                    add_special_tokens=True, return_tensors="pt",
                )
                mask = encoded.attention_mask.to(device)
                hidden = encoder(encoded.input_ids.to(device), attention_mask=mask)[0]
                hidden_tail = hidden[:, -(length - 1) :] if length > 1 else hidden[:, :0]
                mask_tail = mask[:, -(length - 1) :] if length > 1 else mask[:, :0]
                selected_hidden = torch.cat([hidden[:, :1], hidden_tail], dim=1)
                selected_mask = torch.cat([mask[:, :1], mask_tail], dim=1)
                embeddings.append(selected_hidden.to(torch.float16).cpu())
                masks.append(selected_mask.bool().cpu())
    finally:
        del encoder
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(embeddings), torch.cat(masks)


def ensure_prompt_features(dataset: CompiledDataset, config: RefinerConfig) -> PromptFeatures:
    recipe = dataset.metadata["recipe"]
    run = load_run_config(Path(recipe["run_config"]))
    records = list(dataset.assets())
    asset_ids = [asset.record["asset_id"] for asset in records]
    prompts = [asset.record["prompt_metadata"]["prompt"] for asset in records]
    identity: dict[str, Any] = {
        "format": FEATURE_FORMAT,
        "dataset_id": dataset.metadata["dataset_id"],
        "provider": config.prompt_features,
        "model": run.sana.model_id,
        "revision": run.sana.revision,
        "tokens": config.max_prompt_tokens,
        "instruction": SANA_INSTRUCTION,
        "assets": asset_ids,
        "prompts": prompts,
    }
    feature_id = stable_id(identity)
    destination = config.feature_root / FEATURE_FORMAT / feature_id
    if destination.exists():
        return PromptFeatures(destination)
    temporary = destination.with_name(destination.name + ".building")
    temporary.mkdir(parents=True, exist_ok=True)
    if config.prompt_features == "fake":
        embeddings, masks = _fake_features(prompts, config.max_prompt_tokens)
    else:
        embeddings, masks = _sana_features(
            prompts, run.sana.model_id, run.sana.revision, config.max_prompt_tokens,
            config.prompt_batch_size, _device(config.device),
        )
    save_file({"embeddings": embeddings.contiguous(), "masks": masks.contiguous()}, temporary / "features.safetensors")
    (temporary / "index.json").write_text(
        json.dumps({**identity, "feature_id": feature_id, "asset_ids": asset_ids}, sort_keys=True, indent=2) + "\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(destination)
    return PromptFeatures(destination)
