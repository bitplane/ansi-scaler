#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${DATASET_DIR:-}" ]]; then
  echo "Set DATASET_DIR to a compiled dataset directory" >&2
  exit 2
fi
exec .venv/bin/ansi-scaler dataset-validate --dataset-dir "$DATASET_DIR"
