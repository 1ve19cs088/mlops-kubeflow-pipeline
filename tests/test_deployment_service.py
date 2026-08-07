"""
Tests for deployment/service.py — DeploymentService is given an
explicit DeploymentConfig in every test so nothing here depends on
environment variables, a Docker daemon, or a real registry. The
registry query itself (deployment.registry_client) is mocked, per
tests/test_deployment_registry_client.py already covering its own
behavior against a mocked httpx.
"""

from unittest.mock import patch

from deployment.config import DeploymentConfig
from deployment.registry_client import PublishedImage
from deployment.service import DeploymentService, get_deployment_service


def _config(**overrides):
    defaults = dict(
        provider="ghcr",
        registry="ghcr.io",
        repository="1ve19cs088/mlops-kubeflow-pipeline",
        image_name="mlops-kubeflow-pipeline-serving",
        latest_image_tag="not-yet-pushed",
        deployment_target="kind-ai-agent",
        current_deployed_version="Not Deployed",
    )
    defaults.update(overrides)
    return DeploymentConfig(**defaults)


def test_get_status_reports_configured_when_registry_repository_and_image_are_set():
    with patch("deployment.service.get_latest_published_image", return_value=None):
        service = DeploymentService(config=_config())
        status = service.get_status()

    assert status.status == "Configured"
    assert status.registry == "ghcr.io"
    assert status.repository == "1ve19cs088/mlops-kubeflow-pipeline"
    assert status.image_name == "mlops-kubeflow-pipeline-serving"
    assert status.latest_image_tag == "not-yet-pushed"
    assert status.deployment_target == "kind-ai-agent"
    assert status.current_deployed_version == "Not Deployed"


def test_get_status_includes_published_image_details_when_registry_has_one():
    published = PublishedImage(tag="latest", digest="sha256:abc123")

    with patch("deployment.service.get_latest_published_image", return_value=published):
        service = DeploymentService(config=_config())
        status = service.get_status()

    assert status.latest_published_image == (
        "ghcr.io/1ve19cs088/mlops-kubeflow-pipeline-serving:latest"
    )
    assert status.image_digest == "sha256:abc123"
    assert status.current_tag == "latest"


def test_get_status_reports_nothing_published_when_registry_query_finds_nothing():
    with patch("deployment.service.get_latest_published_image", return_value=None):
        service = DeploymentService(config=_config())
        status = service.get_status()

    assert status.latest_published_image is None
    assert status.image_digest is None
    assert status.current_tag is None


def test_get_status_skips_registry_query_entirely_when_not_configured():
    with patch("deployment.service.get_latest_published_image") as mock_query:
        service = DeploymentService(config=_config(repository=""))
        status = service.get_status()

    mock_query.assert_not_called()
    assert status.latest_published_image is None
    assert status.image_digest is None
    assert status.current_tag is None


def test_get_status_reports_not_configured_when_registry_is_missing():
    service = DeploymentService(config=_config(registry=""))

    assert service.get_status().status == "Not Configured"


def test_get_status_reports_not_configured_when_repository_is_missing():
    service = DeploymentService(config=_config(repository=""))

    assert service.get_status().status == "Not Configured"


def test_get_status_reports_not_configured_when_image_name_is_missing():
    service = DeploymentService(config=_config(image_name=""))

    assert service.get_status().status == "Not Configured"


def test_get_deployment_service_returns_a_deployment_service_instance():
    assert isinstance(get_deployment_service(), DeploymentService)
