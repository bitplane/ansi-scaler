#!/usr/bin/env bash
set -euo pipefail

failed=0
for command in cc c++ cargo zstd nvidia-smi; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "missing dependency: ${command}" >&2
        failed=1
    fi
done

if ! .venv/bin/python - <<'PY'
from pathlib import Path
import sysconfig

include = Path(sysconfig.get_path("include"))
header = include / "Python.h"
if not header.is_file():
    raise SystemExit(
        f"missing Python development header: {header}\n"
        "Install the development package matching this interpreter (for example python3.12-dev on Ubuntu), "
        "or reinstall the uv-managed Python and recreate .venv."
    )
PY
then
    failed=1
fi

if command -v nvidia-smi >/dev/null 2>&1 && ! nvidia-smi -L >/dev/null; then
    echo "NVIDIA driver is installed, but no usable CUDA GPU is available" >&2
    failed=1
fi

if ! .venv/bin/python - <<'PY'
try:
    import torch
except Exception as error:
    raise SystemExit(f"PyTorch failed to load: {error}") from error
try:
    import onnxruntime
except Exception as error:
    raise SystemExit(f"ONNX Runtime failed to load after PyTorch: {error}") from error

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
    raise SystemExit("ONNX Runtime CUDAExecutionProvider is unavailable")
print(f"CUDA ready: {torch.cuda.get_device_name(0)}")
PY
then
    failed=1
fi

if (( failed )); then
    echo "dependency check failed; no corpus stages were started" >&2
    exit 1
fi
