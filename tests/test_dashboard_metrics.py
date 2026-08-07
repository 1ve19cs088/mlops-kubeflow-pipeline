"""
Tests for the dashboard's Model Metrics page.

Metrics now come entirely from MlflowRegistryClient — the latest
version of the most recently updated registered model. The confusion
matrix is read from the metrics.json artifact MLflow already logs
(via get_artifact_bytes), since a matrix isn't a scalar MLflow metric.
"""

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from dashboard.main import app
from dashboard.mlflow_client import get_mlflow_client

METRICS = {
    "train_accuracy": 1.0,
    "train_precision": 1.0,
    "train_recall": 1.0,
    "train_f1_score": 1.0,
    "test_accuracy": 0.9667,
    "test_precision": 0.9675,
    "test_recall": 0.9667,
    "test_f1_score": 0.9666,
    "training_duration_seconds": 0.06,
}

METRICS_JSON_ARTIFACT = {
    "test": {
        "confusion_matrix": {
            "labels": ["setosa", "versicolor", "virginica"],
            "matrix": [[10, 0, 0], [0, 9, 1], [0, 0, 10]],
        }
    }
}


def _fake_model_version(version, run_id="run-1"):
    return SimpleNamespace(
        version=version, current_stage="None", run_id=run_id, creation_timestamp=1700000000000
    )


class StubMlflowClient:
    def __init__(self, models=None, versions_by_name=None, metrics_by_run=None, artifacts_by_run=None):
        self._models = models or []
        self._versions_by_name = versions_by_name or {}
        self._metrics_by_run = metrics_by_run or {}
        self._artifacts_by_run = artifacts_by_run or {}

    def get_registered_models(self):
        return self._models

    def get_latest_version(self, name):
        versions = self._versions_by_name.get(name, [])
        return versions[0] if versions else None

    def get_run_metrics(self, run_id):
        return self._metrics_by_run.get(run_id, {})

    def get_artifact_bytes(self, run_id, artifact_path):
        return self._artifacts_by_run[(run_id, artifact_path)]


def test_metrics_page_renders_metrics_and_confusion_matrix_for_latest_version():
    stub = StubMlflowClient(
        models=[SimpleNamespace(name="iris-model", last_updated_timestamp=100)],
        versions_by_name={"iris-model": [_fake_model_version(version=4, run_id="run-4")]},
        metrics_by_run={"run-4": METRICS},
        artifacts_by_run={
            ("run-4", "metrics.json"): json.dumps(METRICS_JSON_ARTIFACT).encode("utf-8")
        },
    )
    app.dependency_overrides[get_mlflow_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/metrics")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "iris-model" in response.text
    assert "0.9667" in response.text
    assert "setosa" in response.text
    assert "versicolor" in response.text
    assert "0.06s" in response.text


def test_metrics_page_shows_empty_state_when_no_models_registered():
    app.dependency_overrides[get_mlflow_client] = lambda: StubMlflowClient(models=[])

    try:
        with TestClient(app) as client:
            response = client.get("/metrics")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "No models registered yet" in response.text


def test_metrics_page_handles_missing_confusion_matrix_artifact_gracefully():
    stub = StubMlflowClient(
        models=[SimpleNamespace(name="iris-model", last_updated_timestamp=100)],
        versions_by_name={"iris-model": [_fake_model_version(version=1, run_id="run-1")]},
        metrics_by_run={"run-1": METRICS},
        artifacts_by_run={},
    )
    app.dependency_overrides[get_mlflow_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/metrics")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "Confusion matrix not available" in response.text
