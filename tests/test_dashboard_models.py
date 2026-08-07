"""
Tests for the dashboard's Model Registry (/models) and model detail
(/models/<name>) pages.

Mocks MlflowRegistryClient via dependency override — same pattern
already used for ApiClient in every other dashboard page's tests.
"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from dashboard.main import app
from dashboard.mlflow_client import get_mlflow_client


def _fake_registered_model(name, creation_timestamp):
    return SimpleNamespace(name=name, creation_timestamp=creation_timestamp)


def _fake_model_version(version, current_stage="None", run_id="run-1"):
    return SimpleNamespace(version=version, current_stage=current_stage, run_id=run_id)


class StubMlflowClient:
    def __init__(self, models, versions_by_name=None, metrics_by_run=None):
        self._models = models
        self._versions_by_name = versions_by_name or {}
        self._metrics_by_run = metrics_by_run or {}

    def get_registered_models(self):
        return self._models

    def get_model_versions(self, name):
        return self._versions_by_name.get(name, [])

    def get_run_metrics(self, run_id):
        return self._metrics_by_run.get(run_id, {})


def test_models_page_lists_registered_models_with_expected_data():
    stub = StubMlflowClient(
        models=[_fake_registered_model("iris-model", creation_timestamp=1700000000000)],
        versions_by_name={
            "iris-model": [
                _fake_model_version(version=5, run_id="run-5"),
                _fake_model_version(version=3, run_id="run-3"),
            ]
        },
        metrics_by_run={"run-5": {"test_accuracy": 0.97}},
    )
    app.dependency_overrides[get_mlflow_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/models")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "iris-model" in response.text
    assert "0.9700" in response.text
    assert "<td>5</td>" in response.text
    assert "<td>2</td>" in response.text
    assert "/models/iris-model" in response.text
    assert "None" in response.text


def test_models_page_shows_empty_state_when_no_models_registered():
    stub = StubMlflowClient(models=[])
    app.dependency_overrides[get_mlflow_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/models")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "No models registered yet" in response.text


def test_models_page_handles_model_with_no_versions_gracefully():
    stub = StubMlflowClient(
        models=[_fake_registered_model("empty-model", creation_timestamp=1700000000000)],
        versions_by_name={"empty-model": []},
    )
    app.dependency_overrides[get_mlflow_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/models")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "empty-model" in response.text
    assert "N/A" in response.text


def test_models_page_highlights_production_stage():
    stub = StubMlflowClient(
        models=[_fake_registered_model("prod-model", creation_timestamp=1700000000000)],
        versions_by_name={
            "prod-model": [_fake_model_version(version=1, current_stage="Production")]
        },
        metrics_by_run={},
    )
    app.dependency_overrides[get_mlflow_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/models")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "bg-success" in response.text
    assert "Production" in response.text


def test_model_detail_page_shows_placeholder_and_real_model_name():
    with TestClient(app) as client:
        response = client.get("/models/iris-RandomForestClassifier")

    assert response.status_code == 200
    assert "iris-RandomForestClassifier" in response.text
    assert "Coming in Stage 3" in response.text
