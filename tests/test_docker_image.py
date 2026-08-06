"""
Tests for the Docker serving image itself — not the application code
(that's covered by tests/test_app_*.py). These build a real image and
run a real container, so they're slow and excluded from the default
test run (see pytest.ini's `addopts = -m "not docker"`).

Run explicitly with: pytest -m docker -v

Assumes the training pipeline has already been run (see README.md's
"Building the serving image" prerequisites) — this tests the Docker
packaging, not the ML pipeline itself.
"""

import json
import subprocess
import time
import urllib.request

import pytest

pytestmark = pytest.mark.docker

IMAGE_TAG = "mlops-kubeflow-pipeline-serving:test"
CONTAINER_NAME = "mlops-serving-pytest"
HOST_PORT = 8001


def _docker(*args, check=True):
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=check
    )


@pytest.fixture(scope="module")
def built_image():
    """Build the serving image once for every test in this module."""

    result = _docker(
        "build", "-f", "docker/Dockerfile.serving", "-t", IMAGE_TAG, ".", check=False
    )

    assert result.returncode == 0, (
        f"docker build failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    yield IMAGE_TAG


@pytest.fixture(scope="module")
def running_container(built_image):
    """Start a container from the built image and wait until it's healthy."""

    _docker("rm", "-f", CONTAINER_NAME, check=False)
    _docker(
        "run", "-d", "--name", CONTAINER_NAME, "-p", f"{HOST_PORT}:8000", built_image
    )

    healthy = False
    for _ in range(30):
        status = _docker(
            "inspect", "--format={{.State.Health.Status}}", CONTAINER_NAME
        ).stdout.strip()
        if status == "healthy":
            healthy = True
            break
        time.sleep(2)

    assert healthy, "container did not become healthy within the wait window"

    yield CONTAINER_NAME

    _docker("stop", CONTAINER_NAME, check=False)
    _docker("rm", CONTAINER_NAME, check=False)


def test_image_builds_successfully(built_image):
    images = _docker(
        "images", built_image, "--format", "{{.Repository}}:{{.Tag}}"
    ).stdout
    assert built_image in images


def test_container_runs_as_non_root(running_container):
    uid = int(_docker("exec", running_container, "id", "-u").stdout.strip())
    assert uid != 0


def test_healthcheck_reports_healthy(running_container):
    status = _docker(
        "inspect", "--format={{.State.Health.Status}}", running_container
    ).stdout.strip()
    assert status == "healthy"


def test_baked_in_artifacts_exist(running_container):
    for path in [
        "/app/model/model.pkl",
        "/app/artifacts/training_report.json",
        "/app/artifacts/metrics.json",
    ]:
        result = _docker("exec", running_container, "test", "-f", path, check=False)
        assert result.returncode == 0, f"missing baked-in artifact: {path}"


def test_dependency_isolation_pipeline_and_test_packages_absent(running_container):
    installed = _docker(
        "exec", running_container, "/opt/venv/bin/pip", "list", "--format=freeze"
    ).stdout.lower()

    for package in ["matplotlib", "kfp", "pyyaml", "pytest", "httpx"]:
        assert package not in installed, f"{package} leaked into the serving image"


def test_api_is_reachable(running_container):
    response = urllib.request.urlopen(f"http://localhost:{HOST_PORT}/v1/health")
    assert response.status == 200


def test_prediction_endpoint_returns_valid_response(running_container):
    body = json.dumps(
        {
            "sepal_length_cm": 5.1,
            "sepal_width_cm": 3.5,
            "petal_length_cm": 1.4,
            "petal_width_cm": 0.2,
        }
    ).encode()

    request = urllib.request.Request(
        f"http://localhost:{HOST_PORT}/v1/predict",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    response = urllib.request.urlopen(request)
    payload = json.loads(response.read())

    assert response.status == 200
    assert payload["prediction"] in ["setosa", "versicolor", "virginica"]


def test_image_metadata_exposes_expected_port(built_image):
    config = json.loads(_docker("inspect", built_image).stdout)[0]
    assert "8000/tcp" in config["Config"]["ExposedPorts"]


def test_image_metadata_cmd_is_exec_form(built_image):
    config = json.loads(_docker("inspect", built_image).stdout)[0]
    assert config["Config"]["Cmd"] == [
        "uvicorn",
        "app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]


def test_filesystem_layout_only_copies_src_config(running_container):
    result = _docker(
        "exec", running_container, "test", "-f", "/app/src/config/settings.py",
        check=False,
    )
    assert result.returncode == 0

    for path in ["/app/src/data", "/app/src/models", "/app/src/features", "/app/src/pipeline"]:
        result = _docker("exec", running_container, "test", "-e", path, check=False)
        assert result.returncode != 0, f"unexpected pipeline-only path leaked in: {path}"


def test_filesystem_layout_owned_by_appuser(running_container):
    owner = _docker("exec", running_container, "stat", "-c", "%U", "/app").stdout.strip()
    assert owner == "appuser"
