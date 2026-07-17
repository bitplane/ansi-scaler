#!/usr/bin/env bash
set -euo pipefail
uv sync --frozen
touch .venv/.installed-dev
rm -f .venv/.installed

