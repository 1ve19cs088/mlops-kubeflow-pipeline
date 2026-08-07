"""
Tests for the real Deploy/Rollback/Promote/Archive dashboard actions:
GET/POST /models/<name>/versions/<version>/actions/<deploy|rollback|promote|archive>

Every mutating call goes through a stub MlflowRegistryClient/
DeploymentService — no real MLflow write, kubectl call, or registry
query happens in this suite.
"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from dashboard.main import app
from dashboard.mlflow_client import get_mlflow_client
from deployment.service import (
    CurrentDeployment,
    DeployabilityCheck,
    DeploymentResult,
    get_deployment_service,
)


def _fake_model_version(version, run_id="run-1", current_stage="None"):
    return SimpleNamespace(
        version=version, current_stage=current_stage, run_id=run_id, creation_timestamp=1700000000000
    )


class StubMlflowClient:
    def __init__(self, versions_by_name=None, git_commits_by_run=None):
        self._versions_by_name = versions_by_name or {}
        self._git_commits_by_run = git_commits_by_run or {}
        self.promoted = []
        self.archived = []
        self.raise_on_promote = None
        self.raise_on_archive = None

    def get_model_versions(self, name):
        return self._versions_by_name.get(name, [])

    def get_git_commit(self, run_id):
        return self._git_commits_by_run.get(run_id)

    def promote_version(self, name, version):
        if self.raise_on_promote:
            raise self.raise_on_promote
        self.promoted.append((name, version))

    def archive_version(self, name, version):
        if self.raise_on_archive:
            raise self.raise_on_archive
        self.archived.append((name, version))


class StubDeploymentService:
    def __init__(self, deployable=True, reason=None, image="ghcr.io/owner/image:abc123", deploy_result=None):
        self._deployable = deployable
        self._reason = reason
        self._image = image
        self._deploy_result = deploy_result
        self.deploy_calls = []

    def resolve_deployment_for_commit(self, git_commit_sha):
        if not git_commit_sha or not self._deployable:
            return DeployabilityCheck(
                deployable=False, image=None, reason=self._reason or "No published image was found."
            )
        return DeployabilityCheck(deployable=True, image=self._image, reason=None)

    def get_current_deployment(self):
        return CurrentDeployment(image=None, tag=None)

    def get_status(self):
        return SimpleNamespace(deployment_target="kind-ai-agent")

    def deploy(self, image_tag, timeout_seconds=120):
        self.deploy_calls.append(image_tag)
        if self._deploy_result is not None:
            return self._deploy_result
        return DeploymentResult(
            success=True,
            image=self._image,
            namespace="mlops-kubeflow-pipeline",
            deployment_name="model-serving",
            duration_seconds=5.0,
            message="Rollout completed successfully.",
        )


def _one_version_client(git_commit="abc123commit"):
    return StubMlflowClient(
        versions_by_name={"iris-model": [_fake_model_version(version=1, run_id="run-1")]},
        git_commits_by_run={"run-1": git_commit} if git_commit else {},
    )


# ---- Deploy ----


def test_deploy_confirm_shows_details_when_deployable():
    app.dependency_overrides[get_mlflow_client] = lambda: _one_version_client()
    app.dependency_overrides[get_deployment_service] = lambda: StubDeploymentService()

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1/actions/deploy")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "Confirm Deploy" in response.text
    assert "iris-model" in response.text
    assert "abc123commit" in response.text
    assert "ghcr.io/owner/image:abc123" in response.text
    assert "kind-ai-agent" in response.text
    assert "Enabled" in response.text  # rollback protection


def test_deploy_confirm_returns_409_and_explains_when_not_deployable():
    app.dependency_overrides[get_mlflow_client] = lambda: _one_version_client(git_commit=None)
    app.dependency_overrides[get_deployment_service] = lambda: StubDeploymentService()

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1/actions/deploy")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 409
    assert "unavailable" in response.text.lower()
    assert "No published image was found" in response.text


def test_deploy_execute_calls_deployment_service_and_shows_success():
    mlflow_stub = _one_version_client()
    deployment_stub = StubDeploymentService()
    app.dependency_overrides[get_mlflow_client] = lambda: mlflow_stub
    app.dependency_overrides[get_deployment_service] = lambda: deployment_stub

    try:
        with TestClient(app) as client:
            response = client.post("/models/iris-model/versions/1/actions/deploy")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "Deploy Result" in response.text
    assert "succeeded" in response.text.lower()
    assert deployment_stub.deploy_calls == ["abc123commit"]


def test_deploy_execute_shows_original_error_and_rollback_status_on_failure():
    failed_result = DeploymentResult(
        success=False,
        image="ghcr.io/owner/image:abc123",
        namespace="mlops-kubeflow-pipeline",
        deployment_name="model-serving",
        duration_seconds=10.0,
        message="Rollout failed or timed out: timed out",
        rolled_back=True,
        rollback_success=True,
        rollback_duration_seconds=2.0,
        rollback_message="Rollback completed successfully.",
        original_error="Rollout failed or timed out: timed out",
    )
    app.dependency_overrides[get_mlflow_client] = lambda: _one_version_client()
    app.dependency_overrides[get_deployment_service] = lambda: StubDeploymentService(
        deploy_result=failed_result
    )

    try:
        with TestClient(app) as client:
            response = client.post("/models/iris-model/versions/1/actions/deploy")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "failed" in response.text.lower()
    assert "Automatic Rollback" in response.text
    assert "Rollout failed or timed out: timed out" in response.text
    assert "Rollback completed successfully." in response.text


def test_deploy_execute_returns_409_if_no_longer_deployable():
    app.dependency_overrides[get_mlflow_client] = lambda: _one_version_client()
    app.dependency_overrides[get_deployment_service] = lambda: StubDeploymentService(deployable=False)

    try:
        with TestClient(app) as client:
            response = client.post("/models/iris-model/versions/1/actions/deploy")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 409


def test_deploy_confirm_404s_for_unknown_version():
    app.dependency_overrides[get_mlflow_client] = lambda: _one_version_client()
    app.dependency_overrides[get_deployment_service] = lambda: StubDeploymentService()

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/99/actions/deploy")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404


# ---- Rollback (mechanically identical to Deploy, different copy) ----


def test_rollback_confirm_shows_current_and_target_version():
    app.dependency_overrides[get_mlflow_client] = lambda: _one_version_client()
    app.dependency_overrides[get_deployment_service] = lambda: StubDeploymentService()

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1/actions/rollback")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "Confirm Rollback" in response.text
    assert "Rollback Target" in response.text
    assert "iris-model v1" in response.text


def test_rollback_execute_calls_deploy_and_never_kubectl_rollout_undo():
    mlflow_stub = _one_version_client()
    deployment_stub = StubDeploymentService()
    app.dependency_overrides[get_mlflow_client] = lambda: mlflow_stub
    app.dependency_overrides[get_deployment_service] = lambda: deployment_stub

    try:
        with TestClient(app) as client:
            response = client.post("/models/iris-model/versions/1/actions/rollback")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "Rollback Result" in response.text
    # Rollback is user-directed: it must go through deploy(), never a
    # bare kubectl rollout undo — the stub only exposes deploy(), so a
    # successful call here proves the route used that path.
    assert deployment_stub.deploy_calls == ["abc123commit"]


def test_rollback_confirm_returns_409_when_target_not_deployable():
    app.dependency_overrides[get_mlflow_client] = lambda: _one_version_client(git_commit=None)
    app.dependency_overrides[get_deployment_service] = lambda: StubDeploymentService()

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1/actions/rollback")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 409


# ---- Promote ----


def test_promote_confirm_shows_current_stage():
    app.dependency_overrides[get_mlflow_client] = lambda: _one_version_client()

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1/actions/promote")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "Confirm Promote" in response.text
    assert "Production" in response.text


def test_promote_execute_calls_mlflow_client_and_shows_success():
    mlflow_stub = _one_version_client()
    app.dependency_overrides[get_mlflow_client] = lambda: mlflow_stub

    try:
        with TestClient(app) as client:
            response = client.post("/models/iris-model/versions/1/actions/promote")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "successfully" in response.text.lower()
    assert "promoted" in response.text.lower()
    assert mlflow_stub.promoted == [("iris-model", "1")]


def test_promote_execute_shows_honest_failure_and_never_fakes_success():
    mlflow_stub = _one_version_client()
    mlflow_stub.raise_on_promote = RuntimeError("MLflow registry is locked")
    app.dependency_overrides[get_mlflow_client] = lambda: mlflow_stub

    try:
        with TestClient(app) as client:
            response = client.post("/models/iris-model/versions/1/actions/promote")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "failed" in response.text.lower()
    assert "MLflow registry is locked" in response.text


# ---- Archive ----


def test_archive_confirm_shows_current_stage():
    app.dependency_overrides[get_mlflow_client] = lambda: _one_version_client()

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1/actions/archive")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "Confirm Archive" in response.text


def test_archive_execute_calls_mlflow_client_and_shows_success():
    mlflow_stub = _one_version_client()
    app.dependency_overrides[get_mlflow_client] = lambda: mlflow_stub

    try:
        with TestClient(app) as client:
            response = client.post("/models/iris-model/versions/1/actions/archive")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "successfully" in response.text.lower()
    assert "archived" in response.text.lower()
    assert mlflow_stub.archived == [("iris-model", "1")]


def test_archive_execute_shows_honest_failure_and_never_fakes_success():
    mlflow_stub = _one_version_client()
    mlflow_stub.raise_on_archive = RuntimeError("MLflow registry is locked")
    app.dependency_overrides[get_mlflow_client] = lambda: mlflow_stub

    try:
        with TestClient(app) as client:
            response = client.post("/models/iris-model/versions/1/actions/archive")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "failed" in response.text.lower()
    assert "MLflow registry is locked" in response.text


def test_promote_confirm_404s_for_unknown_version():
    app.dependency_overrides[get_mlflow_client] = lambda: _one_version_client()

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/99/actions/promote")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
