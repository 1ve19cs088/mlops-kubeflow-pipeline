"""
Model evaluation stage.

Scores a trained model against both the train and test splits,
persists metrics, a confusion matrix, and test-set predictions, and
returns a typed reference to those artifacts.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config.settings import ARTIFACT_DIR
from src.features.feature_engineering import FeatureEngineeringArtifacts
from src.models.train import TrainingArtifacts
from src.utils.logger import get_logger
from src.utils.report_writer import save_json_report

logger = get_logger(__name__)

METRICS_FILENAME = "metrics.json"
CONFUSION_MATRIX_CSV_FILENAME = "confusion_matrix.csv"
CONFUSION_MATRIX_PNG_FILENAME = "confusion_matrix.png"
PREDICTIONS_FILENAME = "predictions.csv"

METRIC_AVERAGING = "macro"


@dataclass(frozen=True)
class EvaluationArtifacts:
    """
    Artifacts produced by the model evaluation stage.

    `metrics` is carried in-memory (same rationale as
    ValidationArtifacts.dataframe: run_pipeline.py needs it immediately
    for a summary, without re-reading the JSON it just wrote) alongside
    paths to everything persisted to disk.
    """

    metrics: dict
    metrics_path: Path
    confusion_matrix_csv_path: Path
    confusion_matrix_png_path: Path
    predictions_path: Path


def _compute_classification_metrics(y_true, y_pred) -> dict:
    """
    Compute aggregate classification metrics for one set of predictions.

    Uses macro averaging: every class contributes equally to
    precision/recall/F1 regardless of how many samples it has. This is
    the right choice for a balanced dataset like Iris; an imbalanced
    dataset would usually call for "weighted" averaging instead. This
    is currently a fixed constant rather than a config option — worth
    promoting to configs/*.yaml once a second, non-balanced dataset
    makes the choice dataset-dependent.
    """

    return {
        "num_samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average=METRIC_AVERAGING, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average=METRIC_AVERAGING, zero_division=0
        ),
        "f1_score": f1_score(y_true, y_pred, average=METRIC_AVERAGING, zero_division=0),
    }


def _compute_confusion_matrix(y_true, y_pred) -> tuple[list[str], list[list[int]]]:
    """
    Compute a confusion matrix with an explicit, stable label ordering.

    Labels are sorted and passed explicitly to sklearn so the row/column
    order is deterministic and independent of prediction order — the
    CSV and PNG artifacts both depend on this same ordering to stay
    interpretable on their own, without needing metrics.json alongside them.
    """

    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    return labels, matrix.tolist()


def _save_confusion_matrix_png(
    labels: list[str], matrix: list[list[int]], path: Path
) -> None:
    """
    Render and persist the confusion matrix as an image.
    """

    display = ConfusionMatrixDisplay(
        confusion_matrix=np.array(matrix), display_labels=labels
    )
    display.plot()
    display.figure_.savefig(path)
    plt.close(display.figure_)


def run(
    training: TrainingArtifacts,
    features: FeatureEngineeringArtifacts,
    config: dict,
) -> EvaluationArtifacts:
    """
    Execute the model evaluation stage.

    Args:
        training: Artifacts produced by the model training stage.
        features: Paths to the persisted train/test splits.
        config: Parsed YAML configuration.

    Returns:
        EvaluationArtifacts referencing the persisted metrics, confusion
        matrix, and predictions.
    """

    logger.info("Starting model evaluation stage...")

    model = joblib.load(training.model_path)

    with open(training.report_path) as file:
        training_report = json.load(file)

    X_train = pd.read_csv(features.x_train_path)
    y_train = pd.read_csv(features.y_train_path).iloc[:, 0]
    X_test = pd.read_csv(features.x_test_path)
    y_test = pd.read_csv(features.y_test_path).iloc[:, 0]

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_metrics = _compute_classification_metrics(y_train, train_predictions)
    test_metrics = _compute_classification_metrics(y_test, test_predictions)

    labels, matrix = _compute_confusion_matrix(y_test, test_predictions)
    test_metrics["confusion_matrix"] = {"labels": labels, "matrix": matrix}

    metrics = {
        "dataset": config["dataset"]["name"],
        "algorithm": training_report["algorithm"],
        "model_version": config["model"]["version"],
        "trained_at": training_report["trained_at"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "train": train_metrics,
        "test": test_metrics,
    }

    metrics_path = save_json_report(ARTIFACT_DIR, METRICS_FILENAME, metrics)

    confusion_matrix_csv_path = ARTIFACT_DIR / CONFUSION_MATRIX_CSV_FILENAME
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        confusion_matrix_csv_path
    )

    confusion_matrix_png_path = ARTIFACT_DIR / CONFUSION_MATRIX_PNG_FILENAME
    _save_confusion_matrix_png(labels, matrix, confusion_matrix_png_path)

    predictions_path = ARTIFACT_DIR / PREDICTIONS_FILENAME
    pd.DataFrame({"y_true": y_test, "y_pred": test_predictions}).to_csv(
        predictions_path, index=False
    )

    logger.info(f"Model evaluation complete: {metrics}")

    return EvaluationArtifacts(
        metrics=metrics,
        metrics_path=metrics_path,
        confusion_matrix_csv_path=confusion_matrix_csv_path,
        confusion_matrix_png_path=confusion_matrix_png_path,
        predictions_path=predictions_path,
    )
