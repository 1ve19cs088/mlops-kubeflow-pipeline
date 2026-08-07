"""
Tests for the dashboard's System Status page.

The page combines a live API health/model check (ApiClient), local
environment detection, and — new in this stage — an MLflow registry
reachability summary (MlflowRegistryClient), stubbed here so no real
HTTP call or MLflow read happens.
"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from dashboard.api_client import ApiUnavailableError, get_api_client
from dashboard.main import app
from dashboard.mlflow_client import get_mlflow_client


class StubApiClient:
    def get_health(self):
        return {"status": "ok"}

    def get_metadata(self):
        return {"algorithm": "RandomForestClassifier"}


class UnavailableApiClient:
    def get_health(self):
        raise ApiUnavailableError("connection refused")

    def get_metadata(self):
        raise ApiUnavailableError("connection refused")


def _fake_model_version(version, run_id="run-1"):
    return SimpleNamespace(
        version=version, current_stage="None", run_id=run_id, creation_timestamp=1700000000000
    )


class StubMlflowClient:
    def __init__(self, models=None, versions_by_name=None):
        self._models = models or []
        self._versions_by_name = versions_by_name or {}

    def get_registered_models(self):
        return self._models

    def get_model_versions(self, name):
        return self._versions_by_name.get(name, [])

    def get_latest_version(self, name):
        versions = self._versions_by_name.get(name, [])
        return versions[0] if versions else None

    def get_run_metrics(self, run_id):
        return {}


class UnreachableMlflowClient:
    def get_registered_models(self):
        raise Exception("database is locked")


def test_status_page_shows_healthy_when_api_and_model_available():
    stub_mlflow = StubMlflowClient(
        models=[SimpleNamespace(name="iris-model", last_updated_timestamp=100)],
        versions_by_name={"iris-model": [_fake_model_version(version=2)]},
    )
    app.dependency_overrides[get_api_client] = lambda: StubApiClient()
    app.dependency_overrides[get_mlflow_client] = lambda: stub_mlflow

    try:
        with TestClient(app) as client:
            response = client.get("/status")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "Healthy" in response.text
    assert "RandomForestClassifier" in response.text
    assert "actions/workflows/ci.yml/badge.svg" in response.text
    assert "MLflow Registry Status" in response.text
    assert "<p class=\"mb-0\">1</p>" in response.text


def test_status_page_degrades_gracefully_when_api_unavailable():
    app.dependency_overrides[get_api_client] = lambda: UnavailableApiClient()
    app.dependency_overrides[get_mlflow_client] = lambda: StubMlflowClient(models=[])

    try:
        with TestClient(app) as client:
            response = client.get("/status")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "Unreachable" in response.text
    assert "Unavailable" in response.text


def test_status_page_shows_mlflow_unreachable_gracefully():
    app.dependency_overrides[get_api_client] = lambda: StubApiClient()
    app.dependency_overrides[get_mlflow_client] = lambda: UnreachableMlflowClient()

    try:
        with TestClient(app) as client:
            response = client.get("/status")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "MLflow registry unreachable" in response.text
    assert "database is locked" in response.text
