#!/usr/bin/env bash
set -euo pipefail
uv run --frozen pytest --cov=ansi_scaler --cov-report=term-missing --cov-report=html

