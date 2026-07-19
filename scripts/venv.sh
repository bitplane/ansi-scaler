#!/usr/bin/env bash
set -euo pipefail
uv venv --clear --managed-python --python 3.12 .venv
touch .venv/.created
