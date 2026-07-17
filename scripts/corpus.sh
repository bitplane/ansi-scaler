#!/usr/bin/env bash
set -euo pipefail
uv run --frozen ansi-scaler run --through lod --run-config "$1"
