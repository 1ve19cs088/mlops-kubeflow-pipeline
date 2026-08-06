"""
Model training stage.

Loads the train split produced by feature engineering, fits the
algorithm declared in configuration, and persists the trained model
plus a training report.
"""

import inspect
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier

from src.config.settings import ARTIFACT_DIR, MODEL_DIR
from src.features.feature_engineering import FeatureEngineeringArtifacts
from src.utils.logger import get_logger
from src.utils.report_writer import save_json_report

logger = get_logger(__name__)

MODEL_FILENAME = "model.pkl"
TRAINING_REPORT_FILENAME = "training_report.json"

ALGORITHM_REGISTRY = {
    "RandomForestClassifier": RandomForestClassifier,
}


@dataclass(frozen=True)
class TrainingArtifacts:
    """
    Artifacts produced by the model training stage.

    Mirrors a Kubeflow Pipeline component's Output[Model] +
    Output[Metrics] parameters: downstream stages (evaluation, serving)
    depend on these paths, not on hardcoded filenames.
    """

    model_path: Path
    report_path: Path


def build_model(algorithm: str, random_state: int):
    """
    Instantiate an unfitted estimator for the given algorithm name.

    Args:
        algorithm: Algorithm name as declared in configuration.
        random_state: Random seed, passed through only if the
            algorithm's constructor accepts it (not every estimator
            supports random_state, so this keeps the registry generic
            across future algorithms without per-algorithm branching).

    Returns:
        An unfitted scikit-learn estimator.
    """

    if algorithm not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    model_class = ALGORITHM_REGISTRY[algorithm]
    constructor_params = inspect.signature(model_class.__init__).parameters

    kwargs = {}
    if "random_state" in constructor_params:
        kwargs["random_state"] = random_state

    return model_class(**kwargs)


def run(features: FeatureEngineeringArtifacts, config: dict) -> TrainingArtifacts:
    """
    Execute the model training stage.

    Args:
        features: Paths to the persisted train/test splits.
        config: Parsed YAML configuration.

    Returns:
        TrainingArtifacts referencing the trained model and training report.
    """

    logger.info("Starting model training stage...")

    algorithm = config["model"]["algorithm"]
    random_state = config["training"]["random_state"]
    target_column = config["schema"]["target_column"]

    X_train = pd.read_csv(features.x_train_path)
    y_train = pd.read_csv(features.y_train_path).iloc[:, 0]

    model = build_model(algorithm, random_state)
    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / MODEL_FILENAME
    joblib.dump(model, model_path)

    is_classification = not is_numeric_dtype(y_train)
    class_labels = sorted(y_train.unique().tolist()) if is_classification else []
    framework = type(model).__module__.split(".")[0]

    report = {
        "algorithm": algorithm,
        "framework": framework,
        "model_version": config["model"]["version"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(X_train),
        "num_features": X_train.shape[1],
        "model_path": str(model_path),
        "feature_names": list(X_train.columns),
        "feature_dtypes": {
            column: str(dtype) for column, dtype in X_train.dtypes.items()
        },
        "target_column": target_column,
        "class_labels": class_labels,
    }

    report_path = save_json_report(ARTIFACT_DIR, TRAINING_REPORT_FILENAME, report)

    logger.info(f"Model training complete: {report}")

    return TrainingArtifacts(model_path=model_path, report_path=report_path)
