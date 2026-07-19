#!/usr/bin/env bash
set -euo pipefail

arguments=(gc --run-config "${1:-configs/runs/first.yaml}")
if [[ "${GC_CONFIRM:-0}" == "1" ]]; then
    arguments+=(--confirm)
fi
uv run --frozen ansi-scaler "${arguments[@]}"
