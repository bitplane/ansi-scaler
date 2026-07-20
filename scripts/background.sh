#!/usr/bin/env bash
set -euo pipefail
exec uv run --frozen ansi-scaler background --run-config "$1"
