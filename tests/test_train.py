"""
Tests for the model training stage.
"""

import json
from datetime import datetime

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

import src.models.train as train_module
from src.features.feature_engineering import FeatureEngineeringArtifacts
from src.models.train import TrainingArtifacts, build_model, run

CONFIG = {
    "schema": {"target_column": "species"},
    "model": {"algorithm": "RandomForestClassifier", "version": "v1.0.0"},
    "training": {"random_state": 42},
}


def _write_processed_csvs(tmp_path):
    X_train = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5, 6],
            "feature_b": [6, 5, 4, 3, 2, 1],
        }
    )
    y_train = pd.DataFrame({"species": ["a", "b", "a", "b", "a", "b"]})
    X_test = pd.DataFrame({"feature_a": [7, 8], "feature_b": [0, -1]})
    y_test = pd.DataFrame({"species": ["a", "b"]})

    x_train_path = tmp_path / "X_train.csv"
    x_test_path = tmp_path / "X_test.csv"
    y_train_path = tmp_path / "y_train.csv"
    y_test_path = tmp_path / "y_test.csv"

    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    return FeatureEngineeringArtifacts(
        x_train_path=x_train_path,
        x_test_path=x_test_path,
        y_train_path=y_train_path,
        y_test_path=y_test_path,
    )


def test_build_model_returns_correct_estimator():
    model = build_model("RandomForestClassifier", random_state=42)

    assert isinstance(model, RandomForestClassifier)
    assert model.random_state == 42


def test_build_model_rejects_unsupported_algorithm():
    with pytest.raises(ValueError):
        build_model("NotARealAlgorithm", random_state=42)


def test_build_model_omits_random_state_when_unsupported(monkeypatch):
    monkeypatch.setitem(
        train_module.ALGORITHM_REGISTRY, "LinearRegression", LinearRegression
    )

    model = build_model("LinearRegression", random_state=42)

    assert isinstance(model, LinearRegression)
    assert not hasattr(model, "random_state")


def test_run_returns_artifacts_dataclass(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path / "model")
    monkeypatch.setattr(train_module, "ARTIFACT_DIR", tmp_path / "artifacts")

    features = _write_processed_csvs(tmp_path)

    artifacts = run(features, CONFIG)

    assert isinstance(artifacts, TrainingArtifacts)
    assert artifacts.model_path.exists()
    assert artifacts.report_path.exists()


def test_run_saves_a_usable_model(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path / "model")
    monkeypatch.setattr(train_module, "ARTIFACT_DIR", tmp_path / "artifacts")

    features = _write_processed_csvs(tmp_path)

    artifacts = run(features, CONFIG)

    model = joblib.load(artifacts.model_path)
    X_test = pd.read_csv(features.x_test_path)
    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)


def test_run_writes_correct_report_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path / "model")
    monkeypatch.setattr(train_module, "ARTIFACT_DIR", tmp_path / "artifacts")

    features = _write_processed_csvs(tmp_path)

    artifacts = run(features, CONFIG)

    report = json.loads(artifacts.report_path.read_text())

    assert report["algorithm"] == "RandomForestClassifier"
    assert report["framework"] == "sklearn"
    assert report["model_version"] == "v1.0.0"
    assert report["training_samples"] == 6
    assert report["num_features"] == 2
    assert report["model_path"] == str(artifacts.model_path)
    datetime.fromisoformat(report["trained_at"])

    assert report["feature_names"] == ["feature_a", "feature_b"]
    assert report["feature_dtypes"] == {
        "feature_a": "int64",
        "feature_b": "int64",
    }
    assert report["target_column"] == "species"
    assert report["class_labels"] == ["a", "b"]


def test_run_leaves_class_labels_empty_for_numeric_target(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path / "model")
    monkeypatch.setattr(train_module, "ARTIFACT_DIR", tmp_path / "artifacts")

    X_train = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5, 6],
            "feature_b": [6, 5, 4, 3, 2, 1],
        }
    )
    y_train = pd.DataFrame({"target": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]})
    X_test = pd.DataFrame({"feature_a": [7, 8], "feature_b": [0, -1]})
    y_test = pd.DataFrame({"target": [7.7, 8.8]})

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

    monkeypatch.setitem(
        train_module.ALGORITHM_REGISTRY, "LinearRegression", LinearRegression
    )

    config = {
        "schema": {"target_column": "target"},
        "model": {"algorithm": "LinearRegression", "version": "v1.0.0"},
        "training": {"random_state": 42},
    }

    artifacts = run(features, config)

    report = json.loads(artifacts.report_path.read_text())

    assert report["target_column"] == "target"
    assert report["class_labels"] == []


def test_run_is_reproducible_with_same_random_state(tmp_path, monkeypatch):
    features = _write_processed_csvs(tmp_path)

    monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path / "first_model")
    monkeypatch.setattr(train_module, "ARTIFACT_DIR", tmp_path / "first_artifacts")
    first = run(features, CONFIG)
    first_model = joblib.load(first.model_path)

    monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path / "second_model")
    monkeypatch.setattr(train_module, "ARTIFACT_DIR", tmp_path / "second_artifacts")
    second = run(features, CONFIG)
    second_model = joblib.load(second.model_path)

    X_test = pd.read_csv(features.x_test_path)

    assert list(first_model.predict(X_test)) == list(second_model.predict(X_test))
