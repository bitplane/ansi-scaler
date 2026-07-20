#!/usr/bin/env bash
set -euo pipefail
args=(dataset-plan --recipe "${1:-configs/datasets/first.yaml}")
if [[ -n "${DATASET_LIMIT:-}" ]]; then args+=(--limit "$DATASET_LIMIT"); fi
exec .venv/bin/ansi-scaler "${args[@]}"
