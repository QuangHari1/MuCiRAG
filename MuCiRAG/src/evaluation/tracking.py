"""Local MLflow tracking for reproducible TeleQnA benchmark runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


VALID_TRACKING_MODES = {"disabled", "mlflow"}


def scalar_params(config: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Flatten benchmark config values into MLflow-compatible parameter scalars."""
    values: dict[str, str | int | float | bool] = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool)):
            values[key] = value
        elif value is None:
            values[key] = "null"
        else:
            values[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return values


@dataclass
class ExperimentTracker:
    """Log progress, final metrics, and durable local artifacts to MLflow."""

    run: Any | None = None
    mlflow: Any | None = None

    @property
    def enabled(self) -> bool:
        return self.run is not None and self.mlflow is not None

    def log_progress(self, completed: int, correct: int, evaluated: int) -> None:
        if not self.enabled:
            return
        self.mlflow.log_metrics(
            {
                "benchmark.completed": completed,
                "benchmark.correct": correct,
                "benchmark.evaluated": evaluated,
                "benchmark.accuracy": correct / evaluated if evaluated else 0.0,
            },
            step=completed,
        )

    def log_comparison(self, comparison: dict[str, Any]) -> None:
        """Attach same-question comparison values to the active MLflow run."""
        if not self.enabled:
            return
        metrics = {
            f"comparison.{key}": float(value)
            for key, value in comparison.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if metrics:
            self.mlflow.log_metrics(metrics)
        tags = {
            f"comparison.{key}": str(value)
            for key, value in comparison.items()
            if not isinstance(value, (int, float)) or isinstance(value, bool)
        }
        if tags:
            self.mlflow.set_tags(tags)

    def finish(
        self,
        *,
        status: str,
        completed: int,
        correct: int,
        evaluated: int,
        results_path: Path,
        manifest_path: Path,
        analysis_directory: Path | None = None,
        comparison_path: Path | None = None,
        error: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.mlflow.log_metrics(
            {
                "benchmark.completed": completed,
                "benchmark.correct": correct,
                "benchmark.evaluated": evaluated,
                "benchmark.accuracy": correct / evaluated if evaluated else 0.0,
            }
        )
        tags = {"benchmark.status": status}
        if error:
            tags["benchmark.error"] = error
        self.mlflow.set_tags(tags)
        self._log_file(results_path)
        self._log_file(manifest_path)
        self._log_file(comparison_path)
        if analysis_directory and analysis_directory.is_dir():
            summary = analysis_directory / "summary.md"
            if summary.is_file() and summary.stat().st_mtime >= results_path.stat().st_mtime:
                for analysis_file in sorted(analysis_directory.iterdir()):
                    if analysis_file.is_file() and analysis_file.suffix in {".csv", ".json", ".md"}:
                        self._log_file(analysis_file, artifact_path="analysis")
        self.mlflow.end_run(status="FAILED" if status == "failed" else "FINISHED")
        self.run = None

    def _log_file(self, path: Path | None, artifact_path: str | None = None) -> None:
        if path and path.is_file():
            self.mlflow.log_artifact(str(path), artifact_path=artifact_path)


def start_experiment_tracker(
    *,
    mode: str,
    experiment_name: str,
    tracking_directory: Path,
    output_path: Path,
    config: dict[str, Any],
    mlflow_module: Any | None = None,
    source_fingerprint: str | None = None,
) -> ExperimentTracker:
    """Start a local MLflow run unless tracking is deliberately disabled."""
    if mode not in VALID_TRACKING_MODES:
        raise ValueError(f"Unsupported experiment tracking mode: {mode}")
    if mode == "disabled":
        return ExperimentTracker()
    tracking_directory.mkdir(parents=True, exist_ok=True)
    mlflow = mlflow_module or import_module("mlflow")
    database_path = (tracking_directory / "mlflow.db").resolve()
    artifact_directory = (tracking_directory / "artifacts").resolve()
    artifact_directory.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{database_path}")
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_directory.as_uri())
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=output_path.stem)
    mlflow.log_params(scalar_params(config))
    tags = {
        "benchmark.output_path": str(output_path),
        "tracking.backend": "mlflow-local",
    }
    if source_fingerprint:
        tags["migration.source_fingerprint"] = source_fingerprint
    mlflow.set_tags(tags)
    return ExperimentTracker(run=run, mlflow=mlflow)
