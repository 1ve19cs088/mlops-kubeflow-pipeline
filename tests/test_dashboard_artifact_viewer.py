"""
Tests for the Version Details page's artifact viewer (Stage 4):
JSON/text/image previews on /models/<name>/versions/<version>, and
the /models/<name>/versions/<version>/artifacts/<path> download route.

Mocks MlflowRegistryClient via dependency override — same pattern as
tests/test_dashboard_model_detail.py.
"""

import base64
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from dashboard.main import app
from dashboard.mlflow_client import get_mlflow_client


def _fake_model_version(version, run_id="run-1", creation_timestamp=1700000000000):
    return SimpleNamespace(
        version=version,
        current_stage="None",
        run_id=run_id,
        creation_timestamp=creation_timestamp,
    )


def _fake_artifact(path, is_dir=False, file_size=123):
    return SimpleNamespace(path=path, is_dir=is_dir, file_size=file_size)


class StubMlflowClient:
    def __init__(self, versions_by_name=None, artifacts_by_run=None, bytes_by_path=None):
        self._versions_by_name = versions_by_name or {}
        self._artifacts_by_run = artifacts_by_run or {}
        self._bytes_by_path = bytes_by_path or {}

    def get_model_versions(self, name):
        return self._versions_by_name.get(name, [])

    def get_run_metrics(self, run_id):
        return {}

    def get_run_parameters(self, run_id):
        return {}

    def list_run_artifacts(self, run_id):
        return self._artifacts_by_run.get(run_id, [])

    def get_artifact_bytes(self, run_id, artifact_path):
        return self._bytes_by_path[(run_id, artifact_path)]


def test_version_detail_renders_json_artifact_as_a_table():
    payload = {"algorithm": "RandomForestClassifier", "feature_names": ["a", "b"]}
    stub = StubMlflowClient(
        versions_by_name={"iris-model": [_fake_model_version(version=1)]},
        artifacts_by_run={"run-1": [_fake_artifact("training_report.json")]},
        bytes_by_path={("run-1", "training_report.json"): json.dumps(payload).encode("utf-8")},
    )
    app.dependency_overrides[get_mlflow_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "training_report.json" in response.text
    assert "RandomForestClassifier" in response.text
    assert "feature_names" in response.text


def test_version_detail_renders_text_artifact_in_code_block():
    stub = StubMlflowClient(
        versions_by_name={"iris-model": [_fake_model_version(version=1)]},
        artifacts_by_run={"run-1": [_fake_artifact("classification_report.txt")]},
        bytes_by_path={
            ("run-1", "classification_report.txt"): b"precision  recall  f1-score\nsetosa  1.00  1.00"
        },
    )
    app.dependency_overrides[get_mlflow_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "<pre" in response.text
    assert "precision  recall  f1-score" in response.text


def test_version_detail_renders_image_artifact_inline_as_data_uri():
    png_bytes = b"\x89PNG\r\n\x1a\nfakepngcontent"
    stub = StubMlflowClient(
        versions_by_name={"iris-model": [_fake_model_version(version=1)]},
        artifacts_by_run={"run-1": [_fake_artifact("confusion_matrix.png")]},
        bytes_by_path={("run-1", "confusion_matrix.png"): png_bytes},
    )
    app.dependency_overrides[get_mlflow_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    expected_b64 = base64.b64encode(png_bytes).decode("ascii")
    assert f"data:image/png;base64,{expected_b64}" in response.text


def test_version_detail_shows_download_button_for_unpreviewable_artifact():
    stub = StubMlflowClient(
        versions_by_name={"iris-model": [_fake_model_version(version=1)]},
        artifacts_by_run={"run-1": [_fake_artifact("predictions.csv")]},
        bytes_by_path={},
    )
    app.dependency_overrides[get_mlflow_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "No preview available for this file" in response.text
    assert "/models/iris-model/versions/1/artifacts/predictions.csv" in response.text


def test_version_detail_skips_directory_artifacts():
    stub = StubMlflowClient(
        versions_by_name={"iris-model": [_fake_model_version(version=1)]},
        artifacts_by_run={"run-1": [_fake_artifact("model", is_dir=True)]},
    )
    app.dependency_overrides[get_mlflow_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "No artifacts logged for this run." in response.text


def test_version_detail_shows_empty_state_when_no_artifacts_exist():
    stub = StubMlflowClient(
        versions_by_name={"iris-model": [_fake_model_version(version=1)]},
        artifacts_by_run={"run-1": []},
    )
    app.dependency_overrides[get_mlflow_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "No artifacts logged for this run." in response.text


def test_version_detail_falls_back_to_download_when_json_parse_fails():
    stub = StubMlflowClient(
        versions_by_name={"iris-model": [_fake_model_version(version=1)]},
        artifacts_by_run={"run-1": [_fake_artifact("corrupt.json")]},
        bytes_by_path={("run-1", "corrupt.json"): b"not valid json"},
    )
    app.dependency_overrides[get_mlflow_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "No preview available for this file" in response.text


def test_download_artifact_route_streams_raw_bytes_with_attachment_header():
    stub = StubMlflowClient(
        versions_by_name={"iris-model": [_fake_model_version(version=1)]},
        bytes_by_path={("run-1", "predictions.csv"): b"a,b\n1,2\n"},
    )
    app.dependency_overrides[get_mlflow_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/1/artifacts/predictions.csv")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.content == b"a,b\n1,2\n"
    assert "attachment" in response.headers["content-disposition"]
    assert "predictions.csv" in response.headers["content-disposition"]


def test_download_artifact_route_404s_for_unknown_version():
    stub = StubMlflowClient(versions_by_name={"iris-model": [_fake_model_version(version=1)]})
    app.dependency_overrides[get_mlflow_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/models/iris-model/versions/99/artifacts/predictions.csv")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
