#!/usr/bin/env bash
set -euo pipefail

uv run --frozen ansi-scaler classify --run-config "${1:-configs/runs/first.yaml}"
