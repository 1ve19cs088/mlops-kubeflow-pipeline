"""
Tests for deployment/service.py — DeploymentService is given an
explicit DeploymentConfig in every test so nothing here depends on
environment variables, a Docker daemon, or a real registry.
"""

from deployment.config import DeploymentConfig
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
    service = DeploymentService(config=_config())

    status = service.get_status()

    assert status.status == "Configured"
    assert status.registry == "ghcr.io"
    assert status.repository == "1ve19cs088/mlops-kubeflow-pipeline"
    assert status.image_name == "mlops-kubeflow-pipeline-serving"
    assert status.latest_image_tag == "not-yet-pushed"
    assert status.deployment_target == "kind-ai-agent"
    assert status.current_deployed_version == "Not Deployed"


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
