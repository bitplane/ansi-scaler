#!/usr/bin/env bash
set -euo pipefail
arguments=(background --run-config "$1")
if [[ "${RETRY_ERRORS:-0}" == "1" ]]; then
    arguments+=(--retry-errors)
fi
exec uv run --frozen ansi-scaler "${arguments[@]}"
