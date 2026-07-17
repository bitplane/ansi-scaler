#!/usr/bin/env bash
set -euo pipefail
uv run --frozen ansi-scaler prompts build --run-config "$1"

