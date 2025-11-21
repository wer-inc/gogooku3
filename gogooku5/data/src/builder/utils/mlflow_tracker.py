"""MLflow tracking helper shared across dataset builder and training scripts."""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional


class MLflowTracker:
    """Lightweight wrapper that standardises MLflow logging."""

    def __init__(
        self,
        *,
        enabled: bool,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.experiment_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME")
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self._git_sha = self._resolve_git_sha()
        self._hostname = socket.gethostname()

    @classmethod
    def from_settings(cls, settings) -> "MLflowTracker":
        """Build tracker from DatasetBuilderSettings (best effort)."""

        enabled = bool(getattr(settings, "enable_mlflow_logging", False))
        experiment = getattr(settings, "mlflow_experiment_name", None)
        tracking_uri = getattr(settings, "mlflow_tracking_uri", None)
        return cls(enabled=enabled, experiment_name=experiment, tracking_uri=tracking_uri)

    @staticmethod
    def _resolve_git_sha() -> Optional[str]:
        for var in ("GIT_COMMIT", "COMMIT_SHA", "GITHUB_SHA"):
            value = os.getenv(var)
            if value:
                return value
        return None

    def _apply_base_tags(
        self, stage: str, dagster_run_id: Optional[str], extra: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        tags: Dict[str, str] = {
            "stage": stage,
            "host": self._hostname,
        }
        if dagster_run_id:
            tags["dagster_run_id"] = dagster_run_id
        if self._git_sha:
            tags["git_sha"] = self._git_sha
        if extra:
            tags.update({k: v for k, v in extra.items() if v is not None})
        return tags

    @contextmanager
    def start_run(
        self,
        *,
        stage: str,
        dagster_run_id: Optional[str],
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Start an MLflow run if enabled."""

        if not self.enabled:
            yield None
            return

        import mlflow

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)

        run_name = f"{stage}-{dagster_run_id}" if dagster_run_id else f"{stage}-{datetime.utcnow():%Y%m%d%H%M%S}"
        run = mlflow.start_run(run_name=run_name)
        mlflow.set_tags(self._apply_base_tags(stage, dagster_run_id, tags))
        if params:
            self.log_params(params)

        try:
            yield run
        finally:
            mlflow.end_run()

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self.enabled or not params:
            return
        import mlflow

        flat_params = {k: self._stringify_param(v) for k, v in params.items()}
        mlflow.log_params(flat_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self.enabled or not metrics:
            return
        import mlflow

        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        if not self.enabled or not path:
            return
        import mlflow

        if artifact_path:
            mlflow.log_artifact(path, artifact_path=artifact_path)
        else:
            mlflow.log_artifact(path)

    def set_tags(self, tags: Dict[str, Optional[str]]) -> None:
        if not self.enabled or not tags:
            return
        import mlflow

        mlflow.set_tags({k: v for k, v in tags.items() if v})

    @staticmethod
    def _stringify_param(value: Any) -> str:
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        if value is None:
            return "None"
        return str(value)
