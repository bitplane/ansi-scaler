#!/usr/bin/env bash
set -euo pipefail
uv run --frozen ansi-scaler generate --run-config "$1"

