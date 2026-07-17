#!/usr/bin/env bash
set -euo pipefail
uv run --frozen ansi-scaler lod --run-config "$1"

