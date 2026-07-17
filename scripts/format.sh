#!/usr/bin/env bash
set -euo pipefail
uv run --frozen ruff format .
uv run --frozen ruff check --fix .

