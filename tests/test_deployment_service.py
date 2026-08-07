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


def test_resolve_deployment_for_commit_is_not_deployable_when_commit_is_none():
    service = DeploymentService(config=_config())

    result = service.resolve_deployment_for_commit(None)

    assert result.deployable is False
    assert result.image is None
    assert "no recorded git commit" in result.reason


def test_resolve_deployment_for_commit_is_not_deployable_when_no_image_published_for_it():
    with patch("deployment.service.get_published_image", return_value=None) as mock_query:
        service = DeploymentService(config=_config())
        result = service.resolve_deployment_for_commit("abc123commit")

    assert result.deployable is False
    assert result.image is None
    assert "No published image was found for commit abc123commit" in result.reason
    mock_query.assert_called_once_with(service._config, "abc123commit")


def test_resolve_deployment_for_commit_is_deployable_when_image_exists():
    published = PublishedImage(tag="abc123commit", digest="sha256:def456")

    with patch("deployment.service.get_published_image", return_value=published):
        service = DeploymentService(config=_config())
        result = service.resolve_deployment_for_commit("abc123commit")

    assert result.deployable is True
    assert result.reason is None
    assert result.image == "ghcr.io/1ve19cs088/mlops-kubeflow-pipeline-serving:abc123commit"


def test_resolve_deployment_for_commit_never_falls_back_to_latest():
    # Even if "latest" happens to be published, resolving a specific
    # (unpublished) commit must never silently substitute it.
    with patch("deployment.service.get_published_image", return_value=None) as mock_query:
        service = DeploymentService(config=_config())
        result = service.resolve_deployment_for_commit("some-other-commit")

    assert result.deployable is False
    mock_query.assert_called_once_with(service._config, "some-other-commit")


def _mock_bundle():
    return ("<manifest yaml>", "model-serving", "mlops-kubeflow-pipeline")


# Every deploy()-path test below mocks apply_manifest, wait_for_rollout,
# AND rollout_undo together — deploy() may call any of the three, and
# leaving even one real would let a "unit" test shell out to a real
# kubectl against whatever cluster context happens to be active. (This
# is exactly the mistake that slipped through initially: an earlier
# version of these tests left rollout_undo unmocked, and because
# _mock_bundle() returns this project's real Deployment name/namespace,
# running the suite issued a genuine `kubectl rollout undo` against the
# live kind-ai-agent cluster. Every test here now patches all three.)


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
        patch("deployment.service.rollout_undo") as mock_undo,
    ):
        service = DeploymentService(config=_config())
        result = service.deploy("abc123")

    assert result.success is True
    assert result.image == "ghcr.io/1ve19cs088/mlops-kubeflow-pipeline-serving:abc123"
    assert result.namespace == "mlops-kubeflow-pipeline"
    assert result.deployment_name == "model-serving"
    assert result.duration_seconds >= 0
    assert "successfully rolled out" in result.message
    assert result.rolled_back is False
    assert result.rollback_success is None
    assert result.rollback_duration_seconds is None
    assert result.rollback_message is None
    assert result.original_error is None
    mock_apply.assert_called_once_with("<manifest yaml>")
    mock_rollout.assert_called_once_with("model-serving", "mlops-kubeflow-pipeline", 120)
    mock_undo.assert_not_called()


def test_deploy_reports_failure_and_skips_rollout_and_rollback_when_apply_fails():
    with (
        patch("deployment.service.build_manifest_bundle", return_value=_mock_bundle()),
        patch(
            "deployment.service.apply_manifest",
            return_value=KubectlResult(success=False, output="error: unable to parse"),
        ),
        patch("deployment.service.wait_for_rollout") as mock_rollout,
        patch("deployment.service.rollout_undo") as mock_undo,
    ):
        service = DeploymentService(config=_config())
        result = service.deploy("bad-tag")

    assert result.success is False
    assert "kubectl apply failed" in result.message
    assert "unable to parse" in result.message
    assert result.original_error == result.message
    assert result.rolled_back is False
    assert result.rollback_success is None
    mock_rollout.assert_not_called()
    mock_undo.assert_not_called()


def test_deploy_automatically_rolls_back_and_reports_success_when_rollout_fails():
    with (
        patch("deployment.service.build_manifest_bundle", return_value=_mock_bundle()),
        patch(
            "deployment.service.apply_manifest",
            return_value=KubectlResult(success=True, output="configured"),
        ),
        patch(
            "deployment.service.wait_for_rollout",
            side_effect=[
                KubectlResult(success=False, output="error: timed out waiting for the condition"),
                KubectlResult(success=True, output="deployment \"model-serving\" successfully rolled out"),
            ],
        ) as mock_rollout,
        patch(
            "deployment.service.rollout_undo",
            return_value=KubectlResult(success=True, output="deployment.apps/model-serving rolled back"),
        ) as mock_undo,
    ):
        service = DeploymentService(config=_config())
        result = service.deploy("broken-tag")

    assert result.success is False
    assert "Rollout failed or timed out" in result.message
    assert "timed out waiting for the condition" in result.message
    assert result.original_error == result.message
    assert result.rolled_back is True
    assert result.rollback_success is True
    assert result.rollback_duration_seconds >= 0
    # rollback_message reflects the *rollback rollout's* confirmation
    # (the second wait_for_rollout call), not the undo command's own
    # output — that distinction matters for anyone reading this field.
    assert "successfully rolled out" in result.rollback_message
    mock_undo.assert_called_once_with("model-serving", "mlops-kubeflow-pipeline")
    assert mock_rollout.call_count == 2
    mock_rollout.assert_any_call("model-serving", "mlops-kubeflow-pipeline", 120)


def test_deploy_reports_rollback_failure_honestly_when_undo_command_itself_fails():
    with (
        patch("deployment.service.build_manifest_bundle", return_value=_mock_bundle()),
        patch(
            "deployment.service.apply_manifest",
            return_value=KubectlResult(success=True, output="configured"),
        ),
        patch(
            "deployment.service.wait_for_rollout",
            return_value=KubectlResult(success=False, output="error: timed out waiting for the condition"),
        ) as mock_rollout,
        patch(
            "deployment.service.rollout_undo",
            return_value=KubectlResult(success=False, output="error: no rollout history found"),
        ),
    ):
        service = DeploymentService(config=_config())
        result = service.deploy("broken-tag")

    assert result.success is False
    assert result.rolled_back is False
    assert result.rollback_success is False
    assert "kubectl rollout undo failed" in result.rollback_message
    assert "no rollout history found" in result.rollback_message
    assert "Rollout failed or timed out" in result.original_error
    # The rollback command itself failed, so there's no reverted
    # rollout to wait for — wait_for_rollout is only ever called once
    # (the original, failed deploy).
    assert mock_rollout.call_count == 1


def test_deploy_reports_rollback_failure_honestly_when_rollback_rollout_itself_fails():
    with (
        patch("deployment.service.build_manifest_bundle", return_value=_mock_bundle()),
        patch(
            "deployment.service.apply_manifest",
            return_value=KubectlResult(success=True, output="configured"),
        ),
        patch(
            "deployment.service.wait_for_rollout",
            side_effect=[
                KubectlResult(success=False, output="error: timed out waiting for the condition"),
                KubectlResult(success=False, output="error: timed out waiting for the condition"),
            ],
        ) as mock_rollout,
        patch(
            "deployment.service.rollout_undo",
            return_value=KubectlResult(success=True, output="deployment.apps/model-serving rolled back"),
        ),
    ):
        service = DeploymentService(config=_config())
        result = service.deploy("broken-tag")

    assert result.success is False
    assert result.rolled_back is False
    assert result.rollback_success is False
    assert "Rollback rollout failed or timed out" in result.rollback_message
    assert "Rollout failed or timed out" in result.original_error
    assert mock_rollout.call_count == 2


def test_deploy_passes_through_a_custom_timeout_to_both_rollout_waits():
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
        patch("deployment.service.rollout_undo") as mock_undo,
    ):
        service = DeploymentService(config=_config())
        service.deploy("abc123", timeout_seconds=30)

    mock_rollout.assert_called_once_with("model-serving", "mlops-kubeflow-pipeline", 30)
    mock_undo.assert_not_called()


def test_get_current_deployment_reads_the_live_image_and_splits_the_tag():
    with (
        patch(
            "deployment.service.get_deployment_identity",
            return_value=("model-serving", "mlops-kubeflow-pipeline"),
        ),
        patch(
            "deployment.service.get_deployment_image",
            return_value="ghcr.io/1ve19cs088/mlops-kubeflow-pipeline-serving:abc123",
        ) as mock_image,
    ):
        service = DeploymentService(config=_config())
        current = service.get_current_deployment()

    assert current.image == "ghcr.io/1ve19cs088/mlops-kubeflow-pipeline-serving:abc123"
    assert current.tag == "abc123"
    mock_image.assert_called_once_with("model-serving", "mlops-kubeflow-pipeline")


def test_get_current_deployment_reports_none_when_cluster_unreachable():
    with (
        patch(
            "deployment.service.get_deployment_identity",
            return_value=("model-serving", "mlops-kubeflow-pipeline"),
        ),
        patch("deployment.service.get_deployment_image", return_value=None),
    ):
        service = DeploymentService(config=_config())
        current = service.get_current_deployment()

    assert current.image is None
    assert current.tag is None


def test_deployment_history_is_empty_until_first_deploy_call():
    service = DeploymentService(config=_config())

    assert service.get_deployment_history() == []


def test_deploy_records_a_history_entry_newest_first():
    with (
        patch("deployment.service.build_manifest_bundle", return_value=_mock_bundle()),
        patch(
            "deployment.service.apply_manifest",
            return_value=KubectlResult(success=True, output="configured"),
        ),
        patch(
            "deployment.service.wait_for_rollout",
            return_value=KubectlResult(success=True, output="successfully rolled out"),
        ),
        patch("deployment.service.rollout_undo"),
    ):
        service = DeploymentService(config=_config())
        service.deploy("first-tag")
        service.deploy("second-tag")

    history = service.get_deployment_history()

    assert len(history) == 2
    assert history[0].tag == "second-tag"
    assert history[1].tag == "first-tag"
    assert history[0].success is True
    assert history[0].rolled_back is False
    assert history[0].timestamp


def test_deploy_records_history_even_when_it_fails_and_rolls_back():
    with (
        patch("deployment.service.build_manifest_bundle", return_value=_mock_bundle()),
        patch(
            "deployment.service.apply_manifest",
            return_value=KubectlResult(success=True, output="configured"),
        ),
        patch(
            "deployment.service.wait_for_rollout",
            side_effect=[
                KubectlResult(success=False, output="timed out"),
                KubectlResult(success=True, output="successfully rolled out"),
            ],
        ),
        patch(
            "deployment.service.rollout_undo",
            return_value=KubectlResult(success=True, output="rolled back"),
        ),
    ):
        service = DeploymentService(config=_config())
        service.deploy("broken-tag")

    history = service.get_deployment_history()

    assert len(history) == 1
    assert history[0].success is False
    assert history[0].rolled_back is True


def test_get_deployment_service_returns_the_same_shared_instance_every_time():
    first = get_deployment_service()
    second = get_deployment_service()

    assert first is second


def test_a_manually_constructed_deployment_service_has_its_own_independent_history():
    with (
        patch("deployment.service.build_manifest_bundle", return_value=_mock_bundle()),
        patch(
            "deployment.service.apply_manifest",
            return_value=KubectlResult(success=True, output="configured"),
        ),
        patch(
            "deployment.service.wait_for_rollout",
            return_value=KubectlResult(success=True, output="successfully rolled out"),
        ),
        patch("deployment.service.rollout_undo"),
    ):
        standalone = DeploymentService(config=_config())
        standalone.deploy("some-tag")

    assert len(standalone.get_deployment_history()) == 1
    # The shared singleton (used by real dashboard requests) is
    # untouched by a standalone instance's deploy() call.
    assert get_deployment_service().get_deployment_history() == []
