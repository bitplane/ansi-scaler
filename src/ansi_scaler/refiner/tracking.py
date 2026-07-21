from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient


def default_tracking_uri(output_root: Path) -> str:
    return f"sqlite:///{(output_root / 'mlflow.db').resolve()}"


def _flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, name))
        elif isinstance(value, (list, tuple)):
            result[name] = json.dumps(value, sort_keys=True)
        elif value is not None:
            result[name] = value
    return result


@dataclass
class RefinerTracker:
    run_id: str
    metadata_path: Path
    log_steps: int

    @classmethod
    def start(
        cls,
        *,
        output_root: Path,
        experiment: str,
        run_dir: Path,
        run_name: str,
        log_steps: int,
        parameters: dict[str, Any],
        tags: dict[str, str],
    ) -> RefinerTracker:
        configured_uri = os.environ.get("MLFLOW_TRACKING_URI")
        tracking_uri = configured_uri or default_tracking_uri(output_root)
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        existing = client.get_experiment_by_name(experiment)
        if existing is None:
            artifact_location = None
            if configured_uri is None:
                artifact_root = (output_root / "mlflow-artifacts" / experiment).resolve()
                artifact_root.mkdir(parents=True, exist_ok=True)
                artifact_location = artifact_root.as_uri()
            experiment_id = client.create_experiment(experiment, artifact_location=artifact_location)
        else:
            experiment_id = existing.experiment_id

        metadata_path = run_dir / "mlflow-run.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata["tracking_uri"] != tracking_uri:
                raise ValueError(
                    f"This training run belongs to MLflow store {metadata['tracking_uri']}, not {tracking_uri}"
                )
            active = mlflow.start_run(run_id=metadata["mlflow_run_id"])
        else:
            active = mlflow.start_run(experiment_id=experiment_id, run_name=run_name, tags=tags)
            metadata_path.write_text(
                json.dumps(
                    {
                        "experiment": experiment,
                        "mlflow_run_id": active.info.run_id,
                        "tracking_uri": tracking_uri,
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            )
        mlflow.log_params(_flatten(parameters))
        return cls(active.info.run_id, metadata_path, log_steps)

    def metrics(self, values: dict[str, float], *, step: int, prefix: str, force: bool = False) -> None:
        if not force and step % self.log_steps:
            return
        mlflow.log_metrics({f"{prefix}/{key}": float(value) for key, value in values.items()}, step=step)

    def artifact(self, path: Path) -> None:
        if path.is_file():
            mlflow.log_artifact(path)

    def finish(self, status: str = "FINISHED") -> None:
        mlflow.end_run(status=status)
