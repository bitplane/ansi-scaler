#!/usr/bin/env bash
set -euo pipefail
uv venv --python 3.12 .venv
touch .venv/.created
