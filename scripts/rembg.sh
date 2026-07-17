#!/usr/bin/env bash
set -euo pipefail
uv run --frozen ansi-scaler rembg --run-config "$1"

