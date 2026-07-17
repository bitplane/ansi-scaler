#!/usr/bin/env bash
set -euo pipefail
uv run --frozen ansi-scaler catalog validate --run-config "$1"

