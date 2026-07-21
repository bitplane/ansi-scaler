#!/usr/bin/env bash
set -euo pipefail

host="${MLFLOW_HOST:-127.0.0.1}"
port="${MLFLOW_PORT:-5000}"
tracking_uri="${MLFLOW_TRACKING_URI:-sqlite:///$(pwd)/data/training/mlflow.db}"

echo "MLflow UI: http://${host}:${port}"
uv run --frozen mlflow ui --backend-store-uri "${tracking_uri}" --host "${host}" --port "${port}"
