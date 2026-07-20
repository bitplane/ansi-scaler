from __future__ import annotations

import gc
from typing import Any

import numpy as np
from PIL import Image

from ansi_scaler.artifacts import artifact_path, atomic_destination
from ansi_scaler.config import RunConfig
from ansi_scaler.identity import sha256_file, stable_id
from ansi_scaler.manifests import read_jsonl, relative_path, resolve_path
from ansi_scaler.reports import contact_sheet
from ansi_scaler.runner import StageInfrastructureError, run_stage


BACKGROUND_CONTRACT = "background-v1"


class BackgroundProcessor:
    def __init__(
        self,
        config: RunConfig,
        session: Any | None = None,
        remove_function: Any | None = None,
        model: Any | None = None,
        *,
        force: bool = False,
    ) -> None:
        self.config = config
        self.settings = config.background
        self.session = session
        self.remove_function = remove_function
        self.model = model
        self.force = force

    def _load_rembg_onnx(self) -> tuple[Any, Any]:
        if self.session is None or self.remove_function is None:
            import torch  # noqa: F401
            import onnxruntime as ort
            from rembg import new_session, remove

            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("rembg-onnx requires ONNX Runtime's CUDAExecutionProvider, but it is unavailable")
            self.session = new_session(
                self.settings.model,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.remove_function = remove
            if self.settings.model_path is None or self.settings.sha256 is None:
                raise ValueError("rembg-onnx requires model_path and sha256")
            model_path = self.settings.model_path.expanduser()
            if not model_path.exists():
                raise FileNotFoundError(f"Background model was not found at {model_path}")
            checksum = sha256_file(model_path)
            if checksum != self.settings.sha256:
                raise ValueError(f"Unexpected background model SHA-256: {checksum}")
        return self.session, self.remove_function

    def _load_lucida(self) -> Any:
        if self.model is None:
            if not self.settings.revision:
                raise ValueError("lucida-transformers requires a pinned model revision")
            import torch
            from transformers import AutoModelForImageSegmentation

            dtype = getattr(torch, self.settings.dtype)
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.settings.model,
                revision=self.settings.revision,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(self.settings.device)
            self.model.eval()
        return self.model

    def _remove_lucida(self, image: Image.Image) -> Image.Image:
        import torch
        from torch.nn.functional import interpolate

        model = self._load_lucida()
        rgb = image.convert("RGB")
        resized = rgb.resize((self.settings.input_size, self.settings.input_size), Image.Resampling.BILINEAR)
        values = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0)
        mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        dtype = getattr(torch, self.settings.dtype)
        tensor = ((tensor - mean) / std).to(device=self.settings.device, dtype=dtype)
        with torch.inference_mode():
            predictions = model(tensor)
            alpha = predictions[-1].sigmoid()
            alpha = interpolate(alpha, size=(rgb.height, rgb.width), mode="bilinear", align_corners=False)
        mask = (alpha[0, 0].float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        result = rgb.convert("RGBA")
        result.putalpha(Image.fromarray(mask, mode="L"))
        return result

    def close(self) -> None:
        had_model = self.session is not None or self.remove_function is not None or self.model is not None
        self.session = None
        self.remove_function = None
        self.model = None
        gc.collect()
        if had_model:
            import torch

            torch.cuda.empty_cache()

    def output_id(self, source: dict[str, Any]) -> str:
        return stable_id(BACKGROUND_CONTRACT, source["id"], self.settings.model_dump(mode="json"))

    def __call__(self, source: dict[str, Any]) -> dict[str, Any]:
        output_id = self.output_id(source)
        source_path = resolve_path(source["artifact"], self.config.data_dir)
        destination = artifact_path(self.config.artifact_dir, "backgrounds", output_id, ".png")
        if self.force:
            destination.unlink(missing_ok=True)
        if not destination.exists():
            image = Image.open(source_path).convert("RGBA")
            try:
                if self.settings.provider == "rembg-onnx":
                    session, remove_function = self._load_rembg_onnx()
                    cutout = remove_function(image, session=session, alpha_matting=False)
                elif self.settings.provider == "lucida-transformers":
                    cutout = self._remove_lucida(image)
                else:  # pragma: no cover - rejected by configuration validation
                    raise ValueError(f"Unsupported background provider: {self.settings.provider}")
            except Exception as error:
                raise StageInfrastructureError(
                    "Background processing could not initialise or run; fix the model, memory, or CUDA error and resume"
                ) from error
            with atomic_destination(destination) as temporary:
                cutout.save(temporary, format="PNG")

        with Image.open(destination) as cutout:
            histogram = cutout.getchannel("A").histogram()
            total = cutout.width * cutout.height
            opaque_fraction = sum(histogram[224:]) / total
            transparent_fraction = sum(histogram[:32]) / total
            soft_fraction = 1.0 - opaque_fraction - transparent_fraction
        return {
            **source,
            "id": output_id,
            "parent_id": source["id"],
            "stage": "background",
            "contract": BACKGROUND_CONTRACT,
            "source_artifact": source["artifact"],
            "artifact": relative_path(destination, self.config.data_dir),
            "background_provider": self.settings.provider,
            "background_model": self.settings.model,
            "background_model_revision": self.settings.revision,
            "background_model_sha256": self.settings.sha256,
            "background_settings": self.settings.model_dump(mode="json"),
            "opaque_fraction": opaque_fraction,
            "transparent_fraction": transparent_fraction,
            "soft_edge_fraction": soft_fraction,
        }


def run_background(
    config: RunConfig,
    *,
    limit: int | None = None,
    force: bool = False,
    retry_errors: bool = False,
    session: Any | None = None,
    remove_function: Any | None = None,
    model: Any | None = None,
) -> tuple[int, int, int]:
    processor = BackgroundProcessor(config, session=session, remove_function=remove_function, model=model, force=force)
    output = config.manifest_dir / "backgrounds.jsonl"
    result = run_stage(
        read_jsonl(config.manifest_dir / "rasters.jsonl"),
        output,
        config.manifest_dir / "backgrounds.errors.jsonl",
        processor,
        processor.output_id,
        limit=limit or config.limit,
        force=force,
        retry_errors=retry_errors,
        stage_name="background",
    )
    paths = [resolve_path(record["artifact"], config.data_dir) for record in read_jsonl(output)]
    contact_sheet(paths[:100], config.run_dir / "reports" / "backgrounds.png")
    return result
