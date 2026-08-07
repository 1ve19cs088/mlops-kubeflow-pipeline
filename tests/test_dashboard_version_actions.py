"""
Tests for the Stage 6 model-lifecycle action placeholders:
GET /models/<name>/versions/<version>/actions/<action_key>

Every one of these must return 501 and explain itself — never a fake
200 pretending the action happened.
"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from dashboard.main import app
from dashboard.mlflow_client import get_mlflow_client


def _fake_model_version(version, run_id="run-1"):
    return SimpleNamespace(
        version=version, current_stage="None", run_id=run_id, creation_timestamp=1700000000000
    )


class StubMlflowClient:
    def __init__(self, versions_by_name=None):
        self._versions_by_name = versions_by_name or {}

    def get_model_versions(self, name):
        return self._versions_by_name.get(name, [])


def _client_with_one_version():
    return StubMlflowClient(
        versions_by_name={"iris-model": [_fake_model_version(version=1)]}
    )


def test_promote_action_returns_501_and_explains_itself():
    app.dependency_overrides[get_mlflow_client] = _client_with_one_version

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1/actions/promote")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 501
    assert "Promote to Production" in response.text
    assert "Not implemented" in response.text
    assert "iris-model" in response.text
    assert "Production" in response.text


def test_archive_action_returns_501_and_explains_itself():
    app.dependency_overrides[get_mlflow_client] = _client_with_one_version

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1/actions/archive")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 501
    assert "Archive Version" in response.text
    assert "Archived" in response.text


def test_rollback_action_returns_501_and_explains_itself():
    app.dependency_overrides[get_mlflow_client] = _client_with_one_version

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1/actions/rollback")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 501
    assert "Rollback to this Version" in response.text


def test_deploy_action_returns_501_and_explains_itself():
    app.dependency_overrides[get_mlflow_client] = _client_with_one_version

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1/actions/deploy")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 501
    assert "Deploy" in response.text
    assert "Kubernetes" in response.text


def test_unknown_action_key_returns_404():
    app.dependency_overrides[get_mlflow_client] = _client_with_one_version

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1/actions/delete-everything")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404


def test_action_on_unknown_version_returns_404():
    app.dependency_overrides[get_mlflow_client] = _client_with_one_version

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/99/actions/promote")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
