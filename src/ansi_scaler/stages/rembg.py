from __future__ import annotations

import gc
from typing import Any

from PIL import Image

from ansi_scaler.artifacts import artifact_path, atomic_destination
from ansi_scaler.config import RunConfig
from ansi_scaler.identity import sha256_file, stable_id
from ansi_scaler.manifests import read_jsonl, relative_path, resolve_path
from ansi_scaler.reports import contact_sheet
from ansi_scaler.runner import run_stage


class BackgroundRemover:
    def __init__(
        self,
        config: RunConfig,
        session: Any | None = None,
        remove_function: Any | None = None,
        *,
        force: bool = False,
    ) -> None:
        self.config = config
        self.settings = config.rembg
        self.session = session
        self.remove_function = remove_function
        self.force = force

    def _load_model(self) -> tuple[Any, Any]:
        if self.session is None or self.remove_function is None:
            # Importing torch first loads the CUDA libraries bundled in the locked
            # environment, allowing ONNX Runtime to resolve them reliably.
            import torch  # noqa: F401
            import onnxruntime as ort
            from rembg import new_session, remove

            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("rembg requires ONNX Runtime's CUDAExecutionProvider, but it is unavailable")
            self.session = new_session(
                self.settings.model,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.remove_function = remove
            model_path = self.settings.model_path.expanduser()
            if not model_path.exists():
                raise FileNotFoundError(f"rembg model was not found at {model_path}")
            checksum = sha256_file(model_path)
            if checksum != self.settings.sha256:
                raise ValueError(f"Unexpected rembg model SHA-256: {checksum}")
        return self.session, self.remove_function

    def close(self) -> None:
        had_model = self.session is not None or self.remove_function is not None
        self.session = None
        self.remove_function = None
        gc.collect()
        if had_model:
            import torch

            torch.cuda.empty_cache()

    def output_id(self, source: dict[str, Any]) -> str:
        return stable_id("rembg-v1", source["id"], self.settings.model_dump(mode="json"))

    def __call__(self, source: dict[str, Any]) -> dict[str, Any]:
        output_id = self.output_id(source)
        source_path = resolve_path(source["artifact"], self.config.data_dir)
        destination = artifact_path(self.config.artifact_dir, "cutouts", output_id, ".png")
        if self.force:
            destination.unlink(missing_ok=True)
        if not destination.exists():
            session, remove_function = self._load_model()
            image = Image.open(source_path).convert("RGBA")
            cutout = remove_function(image, session=session, alpha_matting=False)
            with atomic_destination(destination) as temporary:
                cutout.save(temporary, format="PNG")

        with Image.open(destination) as cutout:
            alpha = cutout.getchannel("A")
            histogram = alpha.histogram()
            total = cutout.width * cutout.height
            opaque_fraction = sum(histogram[224:]) / total
            transparent_fraction = sum(histogram[:32]) / total
            soft_fraction = 1.0 - opaque_fraction - transparent_fraction
        return {
            **source,
            "id": output_id,
            "parent_id": source["id"],
            "stage": "rembg",
            "source_artifact": source["artifact"],
            "artifact": relative_path(destination, self.config.data_dir),
            "rembg_model": self.settings.model,
            "rembg_model_sha256": self.settings.sha256,
            "opaque_fraction": opaque_fraction,
            "transparent_fraction": transparent_fraction,
            "soft_edge_fraction": soft_fraction,
        }


def run_rembg(
    config: RunConfig,
    *,
    limit: int | None = None,
    force: bool = False,
    retry_errors: bool = False,
    session: Any | None = None,
    remove_function: Any | None = None,
) -> tuple[int, int, int]:
    processor = BackgroundRemover(config, session=session, remove_function=remove_function, force=force)
    output = config.manifest_dir / "cutouts.jsonl"
    result = run_stage(
        read_jsonl(config.manifest_dir / "rasters.jsonl"),
        output,
        config.manifest_dir / "cutouts.errors.jsonl",
        processor,
        processor.output_id,
        limit=limit or config.limit,
        force=force,
        retry_errors=retry_errors,
        stage_name="rembg",
    )
    paths = [resolve_path(record["artifact"], config.data_dir) for record in read_jsonl(output)]
    contact_sheet(paths[:100], config.run_dir / "reports" / "cutouts.png")
    return result
