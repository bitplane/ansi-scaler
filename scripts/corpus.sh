#!/usr/bin/env bash
set -euo pipefail
uv run --frozen ansi-scaler run --through pyramid --run-config "$1"
