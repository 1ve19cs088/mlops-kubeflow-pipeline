"""
MLflow experiment tracking and model registration.

Reads the same artifacts src/models/train.py and src/models/evaluate.py
already produce and write to disk — no business logic is duplicated or
changed here; this module only reports what those stages already did.

Every run is automatically tagged with the git commit it trained
under (MLflow's own standard mlflow.source.git.commit tag, via
GitPython, already an installed dependency) — no code in this module
sets that tag itself; MLflow does it natively whenever start_run() is
called from inside a git working directory. This is what lets a
later registered model version be traced back to the exact commit,
and from there to the exact Docker image CI publishes for that
commit. Confirmed empirically before assuming it needed any help:
version 1, registered back in Phase 1 before this comment existed,
already carries this tag.
"""

import json
import os

import joblib
import mlflow
import mlflow.sklearn

from src.config.settings import PROJECT_ROOT
from src.features.feature_engineering import FeatureEngineeringArtifacts
from src.models.evaluate import EvaluationArtifacts
from src.models.train import TrainingArtifacts
from src.utils.logger import get_logger

logger = get_logger(__name__)

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT}/mlflow.db"
)
# mlflow.set_experiment() alone would silently default a *new*
# experiment's artifact storage to a "mlruns/" directory relative to
# the current working directory, regardless of MLFLOW_TRACKING_URI —
# this makes that location explicit and independently configurable.
MLFLOW_ARTIFACT_ROOT = os.environ.get(
    "MLFLOW_ARTIFACT_ROOT", f"file://{PROJECT_ROOT}/mlruns"
)

MODEL_ARTIFACT_NAME = "model"
METRIC_KEYS = ("accuracy", "precision", "recall", "f1_score")


def _get_or_create_experiment(name: str) -> str:
    """
    Returns the experiment ID for `name`, creating it with an explicit
    artifact_location if it doesn't already exist.

    Returns:
        The experiment ID.
    """

    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id

    return mlflow.create_experiment(name, artifact_location=MLFLOW_ARTIFACT_ROOT)


def log_run(
    training: TrainingArtifacts,
    features: FeatureEngineeringArtifacts,
    evaluation: EvaluationArtifacts,
    config: dict,
    training_duration_seconds: float,
) -> str:
    """
    Log one completed pipeline run to MLflow and register the model.

    Args:
        training: Artifacts produced by the training stage.
        features: Paths to the persisted train/test splits (unused
            directly here, but accepted for a stable, self-documenting
            call signature mirroring evaluate.run()'s own).
        evaluation: Artifacts produced by the evaluation stage.
        config: Parsed YAML configuration.
        training_duration_seconds: Wall-clock time spent fitting the
            model, measured by the caller — this module doesn't
            re-measure anything train.py already did.

    Returns:
        The MLflow run ID.
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    dataset_name = config["dataset"]["name"]
    experiment_id = _get_or_create_experiment(dataset_name)

    training_report = json.loads(training.report_path.read_text())

    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.log_param("algorithm", training_report["algorithm"])
        mlflow.log_param("framework", training_report["framework"])
        mlflow.log_param("model_version", training_report["model_version"])
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("target_column", training_report["target_column"])
        mlflow.log_param(
            "feature_names", ",".join(training_report["feature_names"])
        )
        mlflow.log_param("test_size", config["training"]["test_size"])
        mlflow.log_param("random_state", config["training"]["random_state"])

        for split_name in ("train", "test"):
            split_metrics = evaluation.metrics[split_name]
            for key in METRIC_KEYS:
                mlflow.log_metric(f"{split_name}_{key}", split_metrics[key])

        mlflow.log_metric("training_duration_seconds", training_duration_seconds)

        mlflow.log_artifact(str(training.report_path))
        mlflow.log_artifact(str(evaluation.metrics_path))
        mlflow.log_artifact(str(evaluation.confusion_matrix_csv_path))
        mlflow.log_artifact(str(evaluation.confusion_matrix_png_path))
        mlflow.log_artifact(str(evaluation.predictions_path))

        model = joblib.load(training.model_path)
        registered_model_name = f"{dataset_name}-{training_report['algorithm']}"
        mlflow.sklearn.log_model(
            sk_model=model,
            name=MODEL_ARTIFACT_NAME,
            registered_model_name=registered_model_name,
            # MLflow's default serialization format is "skops", which
            # would need another dependency this project doesn't
            # otherwise use. "pickle" matches how train.py already
            # persists this exact model (joblib, which is pickle-based)
            # — no new serialization format introduced, no new
            # dependency needed.
            serialization_format="pickle",
        )

        run_id = run.info.run_id

    logger.info(f"MLflow run logged: {run_id} (experiment={dataset_name})")

    return run_id
