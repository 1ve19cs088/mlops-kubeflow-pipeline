"""
Tests for the dashboard's Home page.

Home shows a live API health check (ApiClient) plus a registry
summary (MlflowRegistryClient) — both are stubbed via dependency
override so no real HTTP call or MLflow read happens here.
"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from dashboard.api_client import ApiUnavailableError, get_api_client
from dashboard.main import app
from dashboard.mlflow_client import get_mlflow_client


class StubApiClient:
    def get_health(self):
        return {"status": "ok"}


class UnavailableApiClient:
    def get_health(self):
        raise ApiUnavailableError("connection refused")


def _fake_model_version(version, current_stage="None", run_id="run-1", creation_timestamp=1700000000000):
    return SimpleNamespace(
        version=version,
        current_stage=current_stage,
        run_id=run_id,
        creation_timestamp=creation_timestamp,
    )


class StubMlflowClient:
    def __init__(self, models=None, versions_by_name=None, metrics_by_run=None):
        self._models = models or []
        self._versions_by_name = versions_by_name or {}
        self._metrics_by_run = metrics_by_run or {}

    def get_registered_models(self):
        return self._models

    def get_model_versions(self, name):
        return self._versions_by_name.get(name, [])

    def get_latest_version(self, name):
        versions = self._versions_by_name.get(name, [])
        return versions[0] if versions else None

    def get_run_metrics(self, run_id):
        return self._metrics_by_run.get(run_id, {})


def test_home_page_shows_registry_summary_for_latest_model():
    stub_mlflow = StubMlflowClient(
        models=[SimpleNamespace(name="iris-model", last_updated_timestamp=200)],
        versions_by_name={
            "iris-model": [
                _fake_model_version(version=3, run_id="run-3", current_stage="Production"),
                _fake_model_version(version=2, run_id="run-2"),
            ]
        },
        metrics_by_run={"run-3": {"test_accuracy": 0.97}},
    )
    app.dependency_overrides[get_api_client] = lambda: StubApiClient()
    app.dependency_overrides[get_mlflow_client] = lambda: stub_mlflow

    try:
        with TestClient(app) as client:
            response = client.get("/")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "iris-model" in response.text
    assert "<p class=\"mb-0\">3</p>" in response.text
    assert "0.9700" in response.text
    assert "bg-success" in response.text
    assert "Production" in response.text
    assert "<p class=\"mb-0\">1</p>" in response.text  # total registered models
    assert "<p class=\"mb-0\">2</p>" in response.text  # total versions


def test_home_page_shows_empty_state_when_no_models_registered():
    app.dependency_overrides[get_api_client] = lambda: StubApiClient()
    app.dependency_overrides[get_mlflow_client] = lambda: StubMlflowClient(models=[])

    try:
        with TestClient(app) as client:
            response = client.get("/")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "No models registered yet" in response.text


def test_home_page_degrades_gracefully_when_api_unavailable():
    app.dependency_overrides[get_api_client] = lambda: UnavailableApiClient()
    app.dependency_overrides[get_mlflow_client] = lambda: StubMlflowClient(models=[])

    try:
        with TestClient(app) as client:
            response = client.get("/")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "API unreachable" in response.text
