"""
Tests for MLflow experiment tracking and model registration.

Uses a real, isolated sqlite-backed MLflow tracking store under
tmp_path — consistent with this project's established testing
philosophy of exercising real behavior against isolated fixtures
rather than mocking, while never touching the project's real
mlflow.db.
"""

import json

import joblib
import mlflow
import pytest
from sklearn.ensemble import RandomForestClassifier

import src.tracking.mlflow_tracking as mlflow_tracking_module
from src.features.feature_engineering import FeatureEngineeringArtifacts
from src.models.evaluate import EvaluationArtifacts
from src.models.train import TrainingArtifacts
from src.tracking.mlflow_tracking import log_run

CONFIG = {
    "dataset": {"name": "test-dataset"},
    "training": {"test_size": 0.2, "random_state": 42},
}


@pytest.fixture(autouse=True)
def isolated_tracking_uri(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mlflow_tracking_module,
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{tmp_path}/mlflow_test.db",
    )
    monkeypatch.setattr(
        mlflow_tracking_module,
        "MLFLOW_ARTIFACT_ROOT",
        f"file://{tmp_path}/mlruns",
    )


def _write_fixture_artifacts(tmp_path):
    X_train = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    y_train = ["a", "b", "a", "b"]

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)

    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)

    training_report = {
        "algorithm": "RandomForestClassifier",
        "framework": "sklearn",
        "model_version": "v1.0.0",
        "trained_at": "2026-01-01T00:00:00+00:00",
        "training_samples": 4,
        "num_features": 2,
        "model_path": str(model_path),
        "feature_names": ["feature_a", "feature_b"],
        "feature_dtypes": {"feature_a": "float64", "feature_b": "float64"},
        "target_column": "species",
        "class_labels": ["a", "b"],
    }
    report_path = tmp_path / "training_report.json"
    report_path.write_text(json.dumps(training_report))

    training = TrainingArtifacts(model_path=model_path, report_path=report_path)

    features = FeatureEngineeringArtifacts(
        x_train_path=tmp_path / "X_train.csv",
        x_test_path=tmp_path / "X_test.csv",
        y_train_path=tmp_path / "y_train.csv",
        y_test_path=tmp_path / "y_test.csv",
    )

    metrics = {
        "dataset": "test-dataset",
        "algorithm": "RandomForestClassifier",
        "model_version": "v1.0.0",
        "trained_at": "2026-01-01T00:00:00+00:00",
        "evaluated_at": "2026-01-01T00:05:00+00:00",
        "train": {
            "num_samples": 4,
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
        },
        "test": {
            "num_samples": 2,
            "accuracy": 0.9,
            "precision": 0.9,
            "recall": 0.9,
            "f1_score": 0.9,
            "confusion_matrix": {"labels": ["a", "b"], "matrix": [[1, 0], [0, 1]]},
        },
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))

    confusion_matrix_csv_path = tmp_path / "confusion_matrix.csv"
    confusion_matrix_csv_path.write_text("a,b\n1,0\n0,1\n")

    confusion_matrix_png_path = tmp_path / "confusion_matrix.png"
    confusion_matrix_png_path.write_bytes(b"not a real png, just needs to exist")

    predictions_path = tmp_path / "predictions.csv"
    predictions_path.write_text("y_true,y_pred\na,a\nb,b\n")

    evaluation = EvaluationArtifacts(
        metrics=metrics,
        metrics_path=metrics_path,
        confusion_matrix_csv_path=confusion_matrix_csv_path,
        confusion_matrix_png_path=confusion_matrix_png_path,
        predictions_path=predictions_path,
    )

    return training, features, evaluation


def test_log_run_logs_params_and_metrics(tmp_path):
    training, features, evaluation = _write_fixture_artifacts(tmp_path)

    run_id = log_run(training, features, evaluation, CONFIG, training_duration_seconds=1.5)

    run = mlflow.get_run(run_id)

    assert run.data.params["algorithm"] == "RandomForestClassifier"
    assert run.data.params["framework"] == "sklearn"
    assert run.data.params["model_version"] == "v1.0.0"
    assert run.data.params["dataset"] == "test-dataset"
    assert run.data.params["target_column"] == "species"
    assert run.data.params["feature_names"] == "feature_a,feature_b"
    assert run.data.params["test_size"] == "0.2"
    assert run.data.params["random_state"] == "42"

    assert run.data.metrics["train_accuracy"] == 1.0
    assert run.data.metrics["test_accuracy"] == 0.9
    assert run.data.metrics["test_precision"] == 0.9
    assert run.data.metrics["test_recall"] == 0.9
    assert run.data.metrics["test_f1_score"] == 0.9
    assert run.data.metrics["training_duration_seconds"] == 1.5


def test_log_run_logs_all_expected_artifacts(tmp_path):
    training, features, evaluation = _write_fixture_artifacts(tmp_path)

    run_id = log_run(training, features, evaluation, CONFIG, training_duration_seconds=1.5)

    client = mlflow.MlflowClient()
    artifact_paths = {
        artifact.path for artifact in client.list_artifacts(run_id)
    }

    assert "training_report.json" in artifact_paths
    assert "metrics.json" in artifact_paths
    assert "confusion_matrix.csv" in artifact_paths
    assert "confusion_matrix.png" in artifact_paths
    assert "predictions.csv" in artifact_paths

    # In MLflow 3.x, mlflow.sklearn.log_model() logs the model as a
    # separate "Logged Model" entity (run.outputs.model_outputs), not
    # nested under the run's plain artifact directory — confirmed by
    # inspecting the actual API behavior rather than assuming the
    # older artifact-subdirectory convention still applies.
    run = client.get_run(run_id)
    assert len(run.outputs.model_outputs) == 1


def test_log_run_registers_the_model(tmp_path):
    training, features, evaluation = _write_fixture_artifacts(tmp_path)

    run_id = log_run(training, features, evaluation, CONFIG, training_duration_seconds=1.5)

    client = mlflow.MlflowClient()
    versions = client.search_model_versions("name='test-dataset-RandomForestClassifier'")

    assert len(versions) == 1
    assert versions[0].run_id == run_id


def test_log_run_creates_separate_runs_per_call(tmp_path):
    training, features, evaluation = _write_fixture_artifacts(tmp_path)

    first_run_id = log_run(training, features, evaluation, CONFIG, training_duration_seconds=1.0)
    second_run_id = log_run(training, features, evaluation, CONFIG, training_duration_seconds=2.0)

    assert first_run_id != second_run_id

    client = mlflow.MlflowClient()
    versions = client.search_model_versions("name='test-dataset-RandomForestClassifier'")
    assert len(versions) == 2
