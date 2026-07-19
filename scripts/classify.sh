#!/usr/bin/env bash
set -euo pipefail

arguments=(classify --run-config "${1:-configs/runs/first.yaml}")
if [[ "${RETRY_ERRORS:-0}" == "1" ]]; then
    arguments+=(--retry-errors)
fi
uv run --frozen ansi-scaler "${arguments[@]}"
