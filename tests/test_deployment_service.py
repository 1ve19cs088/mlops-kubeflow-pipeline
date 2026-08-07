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
from deployment.kubectl_client import KubectlResult
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


def _mock_bundle():
    return ("<manifest yaml>", "model-serving", "mlops-kubeflow-pipeline")


def test_deploy_returns_success_when_apply_and_rollout_both_succeed():
    with (
        patch("deployment.service.build_manifest_bundle", return_value=_mock_bundle()),
        patch(
            "deployment.service.apply_manifest",
            return_value=KubectlResult(success=True, output="deployment.apps/model-serving configured"),
        ) as mock_apply,
        patch(
            "deployment.service.wait_for_rollout",
            return_value=KubectlResult(success=True, output="successfully rolled out"),
        ) as mock_rollout,
    ):
        service = DeploymentService(config=_config())
        result = service.deploy("abc123")

    assert result.success is True
    assert result.image == "ghcr.io/1ve19cs088/mlops-kubeflow-pipeline-serving:abc123"
    assert result.namespace == "mlops-kubeflow-pipeline"
    assert result.deployment_name == "model-serving"
    assert result.duration_seconds >= 0
    assert "successfully rolled out" in result.message
    mock_apply.assert_called_once_with("<manifest yaml>")
    mock_rollout.assert_called_once_with("model-serving", "mlops-kubeflow-pipeline", 120)


def test_deploy_reports_failure_and_skips_rollout_wait_when_apply_fails():
    with (
        patch("deployment.service.build_manifest_bundle", return_value=_mock_bundle()),
        patch(
            "deployment.service.apply_manifest",
            return_value=KubectlResult(success=False, output="error: unable to parse"),
        ),
        patch("deployment.service.wait_for_rollout") as mock_rollout,
    ):
        service = DeploymentService(config=_config())
        result = service.deploy("bad-tag")

    assert result.success is False
    assert "kubectl apply failed" in result.message
    assert "unable to parse" in result.message
    mock_rollout.assert_not_called()


def test_deploy_reports_failure_when_rollout_fails_or_times_out():
    with (
        patch("deployment.service.build_manifest_bundle", return_value=_mock_bundle()),
        patch(
            "deployment.service.apply_manifest",
            return_value=KubectlResult(success=True, output="configured"),
        ),
        patch(
            "deployment.service.wait_for_rollout",
            return_value=KubectlResult(success=False, output="error: timed out waiting for the condition"),
        ),
    ):
        service = DeploymentService(config=_config())
        result = service.deploy("broken-tag")

    assert result.success is False
    assert "Rollout failed or timed out" in result.message
    assert "timed out waiting for the condition" in result.message


def test_deploy_passes_through_a_custom_timeout():
    with (
        patch("deployment.service.build_manifest_bundle", return_value=_mock_bundle()),
        patch(
            "deployment.service.apply_manifest",
            return_value=KubectlResult(success=True, output="configured"),
        ),
        patch(
            "deployment.service.wait_for_rollout",
            return_value=KubectlResult(success=True, output="successfully rolled out"),
        ) as mock_rollout,
    ):
        service = DeploymentService(config=_config())
        service.deploy("abc123", timeout_seconds=30)

    mock_rollout.assert_called_once_with("model-serving", "mlops-kubeflow-pipeline", 30)
