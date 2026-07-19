#!/usr/bin/env bash
set -euo pipefail

arguments=(run --through verify --run-config "$1")
if [[ "${RETRY_ERRORS:-0}" == "1" ]]; then
    arguments+=(--retry-errors)
fi
uv run --frozen ansi-scaler "${arguments[@]}"
