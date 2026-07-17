#!/usr/bin/env bash
set -euo pipefail
uv run --frozen ruff format --check .
uv run --frozen ruff check .

