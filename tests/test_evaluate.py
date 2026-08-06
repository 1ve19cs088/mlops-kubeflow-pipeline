"""
Tests for the model evaluation stage.
"""

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import src.models.evaluate as evaluate_module
from src.features.feature_engineering import FeatureEngineeringArtifacts
from src.models.evaluate import (
    EvaluationArtifacts,
    _compute_classification_metrics,
    _compute_confusion_matrix,
    _save_confusion_matrix_png,
    run,
)
from src.models.train import TrainingArtifacts

CONFIG = {
    "dataset": {"name": "test_dataset"},
    "model": {"algorithm": "RandomForestClassifier", "version": "v9.9.9"},
}


def _write_features(tmp_path):
    X_train = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5, 6, 7, 8],
            "feature_b": [8, 7, 6, 5, 4, 3, 2, 1],
        }
    )
    y_train = pd.DataFrame({"species": ["a", "b", "a", "b", "a", "b", "a", "b"]})
    X_test = pd.DataFrame({"feature_a": [9, 10, 11, 12], "feature_b": [0, -1, -2, -3]})
    y_test = pd.DataFrame({"species": ["a", "b", "a", "b"]})

    x_train_path = tmp_path / "X_train.csv"
    x_test_path = tmp_path / "X_test.csv"
    y_train_path = tmp_path / "y_train.csv"
    y_test_path = tmp_path / "y_test.csv"

    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    features = FeatureEngineeringArtifacts(
        x_train_path=x_train_path,
        x_test_path=x_test_path,
        y_train_path=y_train_path,
        y_test_path=y_test_path,
    )

    return features, X_train, y_train["species"]


def _write_training_artifacts(
    tmp_path,
    X_train,
    y_train,
    algorithm="RandomForestClassifier",
    trained_at="2020-01-01T00:00:00+00:00",
):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)

    report_path = tmp_path / "training_report.json"
    report_path.write_text(
        json.dumps(
            {
                "algorithm": algorithm,
                "trained_at": trained_at,
                "training_samples": len(X_train),
                "num_features": X_train.shape[1],
                "model_path": str(model_path),
            }
        )
    )

    return TrainingArtifacts(model_path=model_path, report_path=report_path)


def test_compute_classification_metrics_on_perfect_predictions():
    y_true = ["a", "b", "a", "b"]
    y_pred = ["a", "b", "a", "b"]

    metrics = _compute_classification_metrics(y_true, y_pred)

    assert metrics["num_samples"] == 4
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0


def test_compute_classification_metrics_on_imperfect_predictions():
    y_true = ["a", "a", "a", "b"]
    y_pred = ["a", "a", "b", "b"]

    metrics = _compute_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.75
    assert 0 < metrics["precision"] < 1
    assert 0 < metrics["recall"] < 1


def test_compute_confusion_matrix_includes_predicted_only_labels():
    y_true = ["a", "a", "b"]
    y_pred = ["a", "c", "b"]  # model predicted "c", which never appears in y_true

    labels, matrix = _compute_confusion_matrix(y_true, y_pred)

    assert labels == ["a", "b", "c"]
    assert len(matrix) == 3
    assert len(matrix[0]) == 3


def test_save_confusion_matrix_png_creates_file(tmp_path):
    path = tmp_path / "cm.png"

    _save_confusion_matrix_png(["a", "b"], [[1, 0], [0, 1]], path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_run_returns_evaluation_artifacts_dataclass(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_module, "ARTIFACT_DIR", tmp_path)

    features, X_train, y_train = _write_features(tmp_path)
    training = _write_training_artifacts(tmp_path, X_train, y_train)

    artifacts = run(training, features, CONFIG)

    assert isinstance(artifacts, EvaluationArtifacts)
    assert artifacts.metrics_path.exists()
    assert artifacts.confusion_matrix_csv_path.exists()
    assert artifacts.confusion_matrix_png_path.exists()
    assert artifacts.predictions_path.exists()


def test_run_metrics_schema_has_required_top_level_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_module, "ARTIFACT_DIR", tmp_path)

    features, X_train, y_train = _write_features(tmp_path)
    training = _write_training_artifacts(tmp_path, X_train, y_train)

    artifacts = run(training, features, CONFIG)

    for key in (
        "dataset",
        "algorithm",
        "model_version",
        "trained_at",
        "evaluated_at",
        "train",
        "test",
    ):
        assert key in artifacts.metrics


def test_run_reads_algorithm_and_trained_at_from_training_report_not_config(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(evaluate_module, "ARTIFACT_DIR", tmp_path)

    features, X_train, y_train = _write_features(tmp_path)
    training = _write_training_artifacts(
        tmp_path,
        X_train,
        y_train,
        algorithm="SomeOtherAlgorithm",
        trained_at="2019-05-05T00:00:00+00:00",
    )

    artifacts = run(training, features, CONFIG)

    assert artifacts.metrics["algorithm"] == "SomeOtherAlgorithm"
    assert artifacts.metrics["trained_at"] == "2019-05-05T00:00:00+00:00"


def test_run_uses_model_version_from_config_not_trained_at(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_module, "ARTIFACT_DIR", tmp_path)

    features, X_train, y_train = _write_features(tmp_path)
    training = _write_training_artifacts(
        tmp_path, X_train, y_train, trained_at="2021-06-06T00:00:00+00:00"
    )

    artifacts = run(training, features, CONFIG)

    assert artifacts.metrics["model_version"] == "v9.9.9"
    assert artifacts.metrics["model_version"] != artifacts.metrics["trained_at"]


def test_run_train_and_test_metrics_have_matching_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_module, "ARTIFACT_DIR", tmp_path)

    features, X_train, y_train = _write_features(tmp_path)
    training = _write_training_artifacts(tmp_path, X_train, y_train)

    artifacts = run(training, features, CONFIG)

    train_keys = set(artifacts.metrics["train"].keys())
    test_keys = set(artifacts.metrics["test"].keys()) - {"confusion_matrix"}

    assert train_keys == test_keys


def test_run_confusion_matrix_only_present_in_test_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_module, "ARTIFACT_DIR", tmp_path)

    features, X_train, y_train = _write_features(tmp_path)
    training = _write_training_artifacts(tmp_path, X_train, y_train)

    artifacts = run(training, features, CONFIG)

    assert "confusion_matrix" not in artifacts.metrics["train"]
    assert "confusion_matrix" in artifacts.metrics["test"]


def test_run_predictions_csv_matches_test_set_size(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_module, "ARTIFACT_DIR", tmp_path)

    features, X_train, y_train = _write_features(tmp_path)
    training = _write_training_artifacts(tmp_path, X_train, y_train)

    artifacts = run(training, features, CONFIG)

    predictions = pd.read_csv(artifacts.predictions_path)
    X_test = pd.read_csv(features.x_test_path)

    assert list(predictions.columns) == ["y_true", "y_pred"]
    assert len(predictions) == len(X_test)


def test_run_confusion_matrix_csv_matches_metrics_json(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_module, "ARTIFACT_DIR", tmp_path)

    features, X_train, y_train = _write_features(tmp_path)
    training = _write_training_artifacts(tmp_path, X_train, y_train)

    artifacts = run(training, features, CONFIG)

    csv_matrix = pd.read_csv(
        artifacts.confusion_matrix_csv_path, index_col=0
    ).values.tolist()
    json_matrix = artifacts.metrics["test"]["confusion_matrix"]["matrix"]

    assert csv_matrix == json_matrix
