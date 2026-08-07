"""
Tests for the dashboard's Model Details page (/models/<name>) and its
per-version drill-down (/models/<name>/versions/<version>).

Mocks MlflowRegistryClient via dependency override — same pattern as
tests/test_dashboard_models.py.
"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from dashboard.main import app
from dashboard.mlflow_client import get_mlflow_client
from deployment.service import DeployabilityCheck, get_deployment_service


def _fake_model_version(version, current_stage="None", run_id="run-1", creation_timestamp=1700000000000):
    return SimpleNamespace(
        version=version,
        current_stage=current_stage,
        run_id=run_id,
        creation_timestamp=creation_timestamp,
    )


class StubMlflowClient:
    def __init__(
        self, versions_by_name=None, metrics_by_run=None, params_by_run=None, git_commits_by_run=None
    ):
        self._versions_by_name = versions_by_name or {}
        self._metrics_by_run = metrics_by_run or {}
        self._params_by_run = params_by_run or {}
        self._git_commits_by_run = git_commits_by_run or {}

    def get_model_versions(self, name):
        return self._versions_by_name.get(name, [])

    def get_run_metrics(self, run_id):
        return self._metrics_by_run.get(run_id, {})

    def get_run_parameters(self, run_id):
        return self._params_by_run.get(run_id, {})

    def get_git_commit(self, run_id):
        return self._git_commits_by_run.get(run_id)

    def list_run_artifacts(self, run_id):
        return []


class StubDeploymentService:
    """
    A fake standing in for DeploymentService in tests that don't care
    about deployability specifically — every commit resolves to "not
    deployable" by default (matching the real service's behavior when
    a version has no recorded commit), unless overridden.
    """

    def __init__(self, deployable_commits=None):
        self._deployable_commits = deployable_commits or {}

    def resolve_deployment_for_commit(self, git_commit_sha):
        if git_commit_sha in self._deployable_commits:
            return DeployabilityCheck(
                deployable=True, image=self._deployable_commits[git_commit_sha], reason=None
            )
        return DeployabilityCheck(
            deployable=False, image=None, reason="No published image was found."
        )


def test_model_detail_lists_every_version_newest_first_with_full_columns():
    stub = StubMlflowClient(
        versions_by_name={
            "iris-model": [
                _fake_model_version(version=2, run_id="run-2", current_stage="Production"),
                _fake_model_version(version=1, run_id="run-1"),
            ]
        },
        metrics_by_run={
            "run-2": {
                "test_accuracy": 0.97,
                "test_precision": 0.96,
                "test_recall": 0.95,
                "test_f1_score": 0.955,
                "training_duration_seconds": 1.23,
            },
        },
        params_by_run={"run-2": {"algorithm": "RandomForestClassifier", "dataset": "iris"}},
        git_commits_by_run={"run-2": "abc123commit"},
    )
    deployment_stub = StubDeploymentService(
        deployable_commits={"abc123commit": "ghcr.io/owner/image:abc123commit"}
    )
    app.dependency_overrides[get_mlflow_client] = lambda: stub
    app.dependency_overrides[get_deployment_service] = lambda: deployment_stub

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    text = response.text
    assert "iris-model" in text
    assert "0.9700" in text
    assert "0.9600" in text
    assert "0.9500" in text
    assert "0.9550" in text
    assert "1.23s" in text
    assert "RandomForestClassifier" in text
    assert "iris" in text
    assert "bg-success" in text
    assert "/models/iris-model/versions/2" in text
    assert "/models/iris-model/versions/1" in text
    # version 1 has no metrics/params stubbed -> handled gracefully
    assert "N/A" in text
    # Promote/Archive are always available.
    assert "/models/iris-model/versions/2/actions/promote" in text
    assert "/models/iris-model/versions/2/actions/archive" in text
    assert "/models/iris-model/versions/1/actions/promote" in text
    assert "/models/iris-model/versions/1/actions/archive" in text
    # Version 2 has a commit with a real published image -> Deploy/Rollback enabled.
    assert "/models/iris-model/versions/2/actions/rollback" in text
    assert "/models/iris-model/versions/2/actions/deploy" in text
    # Version 1 has no recorded git commit -> Deploy/Rollback disabled, never a link.
    assert "/models/iris-model/versions/1/actions/rollback" not in text
    assert "/models/iris-model/versions/1/actions/deploy" not in text
    assert "Deploy (unavailable)" in text
    assert "Rollback (unavailable)" in text


def test_model_detail_shows_empty_state_when_no_versions_exist():
    stub = StubMlflowClient(versions_by_name={"empty-model": []})
    app.dependency_overrides[get_mlflow_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/models/empty-model")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "No versions registered yet" in response.text


def test_version_detail_shows_all_metrics_and_parameters():
    stub = StubMlflowClient(
        versions_by_name={"iris-model": [_fake_model_version(version=1, run_id="run-1")]},
        metrics_by_run={"run-1": {"test_accuracy": 1.0}},
        params_by_run={"run-1": {"algorithm": "RandomForestClassifier"}},
    )
    app.dependency_overrides[get_mlflow_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "iris-model" in response.text
    assert "test_accuracy" in response.text
    assert "1.0" in response.text
    assert "algorithm" in response.text
    assert "RandomForestClassifier" in response.text
    assert "run-1" in response.text


def test_version_detail_returns_404_for_unknown_version():
    stub = StubMlflowClient(versions_by_name={"iris-model": [_fake_model_version(version=1)]})
    app.dependency_overrides[get_mlflow_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/99")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
    assert "not found" in response.text.lower()
