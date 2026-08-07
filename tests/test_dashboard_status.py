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
from deployment.service import (
    CurrentDeployment,
    DeploymentHistoryEntry,
    DeploymentStatus,
    get_deployment_service,
)


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
    def __init__(self, models=None, versions_by_name=None, git_commits_by_run=None):
        self._models = models or []
        self._versions_by_name = versions_by_name or {}
        self._git_commits_by_run = git_commits_by_run or {}

    def get_registered_models(self):
        return self._models

    def get_model_versions(self, name):
        return self._versions_by_name.get(name, [])

    def get_latest_version(self, name):
        versions = self._versions_by_name.get(name, [])
        return versions[0] if versions else None

    def get_run_metrics(self, run_id):
        return {}

    def get_git_commit(self, run_id):
        return self._git_commits_by_run.get(run_id)


class UnreachableMlflowClient:
    def get_registered_models(self):
        raise Exception("database is locked")


class StubDeploymentService:
    """
    A fake standing in for DeploymentService — never constructs a real
    DeploymentConfig or DeploymentService, so get_status() can't ever
    trigger deployment.registry_client's real (network-calling)
    get_latest_published_image.
    """

    def __init__(self, current_deployment=None, history=None, **status_overrides):
        defaults = dict(
            registry="ghcr.io",
            repository="1ve19cs088/mlops-kubeflow-pipeline",
            image_name="mlops-kubeflow-pipeline-serving",
            latest_image_tag="not-yet-pushed",
            deployment_target="kind-ai-agent",
            current_deployed_version="Not Deployed",
            status="Configured",
            latest_published_image=None,
            image_digest=None,
            current_tag=None,
        )
        defaults.update(status_overrides)
        self._status = DeploymentStatus(**defaults)
        self._current_deployment = current_deployment or CurrentDeployment(image=None, tag=None)
        self._history = history or []

    def get_status(self):
        return self._status

    def get_current_deployment(self):
        return self._current_deployment

    def get_deployment_history(self):
        return self._history


def test_status_page_shows_healthy_when_api_and_model_available():
    stub_mlflow = StubMlflowClient(
        models=[SimpleNamespace(name="iris-model", last_updated_timestamp=100)],
        versions_by_name={"iris-model": [_fake_model_version(version=2)]},
    )
    app.dependency_overrides[get_api_client] = lambda: StubApiClient()
    app.dependency_overrides[get_mlflow_client] = lambda: stub_mlflow
    app.dependency_overrides[get_deployment_service] = lambda: StubDeploymentService()

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
    # Stage 6: Deployment Pipeline section, reflecting this real repo's state
    assert "Deployment Pipeline" in response.text
    assert "Docker Build" in response.text
    assert "Future Stage" in response.text
    # Phase 3 Stage 1: Deployment Registry section
    assert "Deployment Registry" in response.text
    assert "ghcr.io" in response.text
    assert "1ve19cs088/mlops-kubeflow-pipeline" in response.text
    assert "not-yet-pushed" in response.text
    assert "kind-ai-agent" in response.text
    assert "Not Deployed" in response.text
    assert "Configured" in response.text
    # Phase 3 Stage 2: not published yet, by default in this stub
    assert "Not Published" in response.text


def test_status_page_shows_published_image_details_when_available():
    app.dependency_overrides[get_api_client] = lambda: StubApiClient()
    app.dependency_overrides[get_mlflow_client] = lambda: StubMlflowClient(models=[])
    app.dependency_overrides[get_deployment_service] = lambda: StubDeploymentService(
        latest_published_image="ghcr.io/1ve19cs088/mlops-kubeflow-pipeline-serving:latest",
        image_digest="sha256:abc123",
        current_tag="latest",
    )

    try:
        with TestClient(app) as client:
            response = client.get("/status")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "ghcr.io/1ve19cs088/mlops-kubeflow-pipeline-serving:latest" in response.text
    assert "sha256:abc123" in response.text
    assert "<p class=\"mb-0\">latest</p>" in response.text


def test_status_page_handles_unconfigured_deployment_registry_gracefully():
    app.dependency_overrides[get_api_client] = lambda: StubApiClient()
    app.dependency_overrides[get_mlflow_client] = lambda: StubMlflowClient(models=[])
    app.dependency_overrides[get_deployment_service] = lambda: StubDeploymentService(
        registry="", repository="", image_name="", status="Not Configured"
    )

    try:
        with TestClient(app) as client:
            response = client.get("/status")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "Deployment Registry" in response.text
    assert response.text.count("Not Configured") >= 3
    assert "Not Published" in response.text


def test_status_page_degrades_gracefully_when_api_unavailable():
    app.dependency_overrides[get_api_client] = lambda: UnavailableApiClient()
    app.dependency_overrides[get_mlflow_client] = lambda: StubMlflowClient(models=[])
    app.dependency_overrides[get_deployment_service] = lambda: StubDeploymentService()

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
    app.dependency_overrides[get_deployment_service] = lambda: StubDeploymentService()

    try:
        with TestClient(app) as client:
            response = client.get("/status")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "MLflow registry unreachable" in response.text
    assert "database is locked" in response.text


def test_status_page_shows_live_deployment_section_with_resolved_mlflow_version():
    stub_mlflow = StubMlflowClient(
        models=[SimpleNamespace(name="iris-model", last_updated_timestamp=100)],
        versions_by_name={"iris-model": [_fake_model_version(version=5, run_id="run-5")]},
        git_commits_by_run={"run-5": "abc123commit"},
    )
    history = [
        DeploymentHistoryEntry(
            timestamp="2026-01-01T00:00:00+00:00",
            image="ghcr.io/1ve19cs088/mlops-kubeflow-pipeline-serving:abc123commit",
            tag="abc123commit",
            success=True,
            rolled_back=False,
            duration_seconds=12.5,
        )
    ]
    deployment_stub = StubDeploymentService(
        current_deployment=CurrentDeployment(
            image="ghcr.io/1ve19cs088/mlops-kubeflow-pipeline-serving:abc123commit",
            tag="abc123commit",
        ),
        history=history,
    )
    app.dependency_overrides[get_api_client] = lambda: StubApiClient()
    app.dependency_overrides[get_mlflow_client] = lambda: stub_mlflow
    app.dependency_overrides[get_deployment_service] = lambda: deployment_stub

    try:
        with TestClient(app) as client:
            response = client.get("/status")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "Live Deployment" in response.text
    assert "ghcr.io/1ve19cs088/mlops-kubeflow-pipeline-serving:abc123commit" in response.text
    assert "abc123commit" in response.text
    assert "iris-model v5" in response.text
    assert "2026-01-01T00:00:00+00:00" in response.text
    assert "12.50s" in response.text


def test_status_page_shows_no_deployments_this_session_when_history_is_empty():
    app.dependency_overrides[get_api_client] = lambda: StubApiClient()
    app.dependency_overrides[get_mlflow_client] = lambda: StubMlflowClient(models=[])
    app.dependency_overrides[get_deployment_service] = lambda: StubDeploymentService()

    try:
        with TestClient(app) as client:
            response = client.get("/status")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "No deployments have been made this session" in response.text
    assert "No deployments this session" in response.text
