#!/usr/bin/env bash
set -euo pipefail
uv sync --frozen --no-dev
touch .venv/.installed
rm -f .venv/.installed-dev

