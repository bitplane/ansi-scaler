#!/usr/bin/env bash
set -euo pipefail
uv venv --managed-python --python 3.12 .venv
touch .venv/.created
