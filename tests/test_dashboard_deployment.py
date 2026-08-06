"""
Tests for dashboard deployment info: reads local kubernetes/ manifests
and safe filesystem markers only — never a live cluster query.
"""

from fastapi.testclient import TestClient

import dashboard.deployment_info as deployment_info_module
from dashboard.deployment_info import get_deployment_info
from dashboard.main import app

NAMESPACE_YAML = """
apiVersion: v1
kind: Namespace
metadata:
  name: mlops-kubeflow-pipeline
"""

DEPLOYMENT_YAML = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: model-serving
          image: mlops-kubeflow-pipeline-serving:v1
"""


def _write_manifests(tmp_path):
    (tmp_path / "namespace.yaml").write_text(NAMESPACE_YAML)
    (tmp_path / "deployment.yaml").write_text(DEPLOYMENT_YAML)
    return tmp_path


def test_get_deployment_info_parses_manifests(tmp_path, monkeypatch):
    _write_manifests(tmp_path)
    monkeypatch.setattr(deployment_info_module, "KUBERNETES_DIR", tmp_path)
    monkeypatch.setattr(
        deployment_info_module, "DOCKER_ENV_MARKER", tmp_path / "no-dockerenv"
    )
    monkeypatch.setattr(
        deployment_info_module,
        "KUBERNETES_SERVICEACCOUNT_MARKER",
        tmp_path / "no-serviceaccount",
    )

    info = get_deployment_info()

    assert info["namespace"] == "mlops-kubeflow-pipeline"
    assert info["deployment_name"] == "model-serving"
    assert info["replica_count"] == 2
    assert info["container_image"] == "mlops-kubeflow-pipeline-serving:v1"
    assert info["docker"] is False
    assert info["kubernetes"] is False
    assert info["environment"] == "local"


def test_get_deployment_info_detects_docker_marker(tmp_path, monkeypatch):
    docker_marker = tmp_path / "dockerenv"
    docker_marker.write_text("")

    monkeypatch.setattr(deployment_info_module, "KUBERNETES_DIR", tmp_path)
    monkeypatch.setattr(deployment_info_module, "DOCKER_ENV_MARKER", docker_marker)
    monkeypatch.setattr(
        deployment_info_module,
        "KUBERNETES_SERVICEACCOUNT_MARKER",
        tmp_path / "no-serviceaccount",
    )

    info = get_deployment_info()

    assert info["docker"] is True
    assert info["environment"] == "docker"


def test_get_deployment_info_degrades_gracefully_without_manifests(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(deployment_info_module, "KUBERNETES_DIR", tmp_path)
    monkeypatch.setattr(
        deployment_info_module, "DOCKER_ENV_MARKER", tmp_path / "no-dockerenv"
    )
    monkeypatch.setattr(
        deployment_info_module,
        "KUBERNETES_SERVICEACCOUNT_MARKER",
        tmp_path / "no-serviceaccount",
    )

    info = get_deployment_info()

    assert info["namespace"] is None
    assert info["deployment_name"] is None
    assert info["replica_count"] is None
    assert info["container_image"] is None


def test_deployment_page_renders():
    with TestClient(app) as client:
        response = client.get("/deployment")

    assert response.status_code == 200
    assert "Deployment" in response.text
