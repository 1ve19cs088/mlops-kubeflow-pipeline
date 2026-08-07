"""
Tests for deployment/kubectl_client.py.

Mocks subprocess.run entirely — no real kubectl invocation, no
cluster, and no Docker daemon are needed to run this suite.
"""

import subprocess
from unittest.mock import MagicMock, patch

from deployment.kubectl_client import apply_manifest, wait_for_rollout


def _completed_process(returncode=0, stdout="", stderr=""):
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


def test_apply_manifest_reports_success_on_zero_exit():
    with patch(
        "subprocess.run", return_value=_completed_process(0, stdout="deployment.apps/model-serving configured\n")
    ) as mock_run:
        result = apply_manifest("apiVersion: v1\nkind: Namespace\n")

    assert result.success is True
    assert "configured" in result.output
    args, kwargs = mock_run.call_args
    assert args[0] == ["kubectl", "apply", "-f", "-"]
    assert kwargs["input"] == "apiVersion: v1\nkind: Namespace\n"


def test_apply_manifest_reports_failure_on_nonzero_exit():
    with patch(
        "subprocess.run",
        return_value=_completed_process(1, stderr="error: unable to parse\n"),
    ):
        result = apply_manifest("not: valid: yaml: at: all")

    assert result.success is False
    assert "unable to parse" in result.output


def test_apply_manifest_handles_kubectl_not_found():
    with patch("subprocess.run", side_effect=FileNotFoundError("kubectl not found")):
        result = apply_manifest("apiVersion: v1\nkind: Namespace\n")

    assert result.success is False
    assert "kubectl not found" in result.output


def test_apply_manifest_handles_timeout():
    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["kubectl"], timeout=30),
    ):
        result = apply_manifest("apiVersion: v1\nkind: Namespace\n")

    assert result.success is False


def test_wait_for_rollout_reports_success():
    with patch(
        "subprocess.run",
        return_value=_completed_process(
            0, stdout="deployment \"model-serving\" successfully rolled out\n"
        ),
    ) as mock_run:
        result = wait_for_rollout("model-serving", "mlops-kubeflow-pipeline", timeout_seconds=60)

    assert result.success is True
    assert "successfully rolled out" in result.output
    args, kwargs = mock_run.call_args
    assert args[0] == [
        "kubectl",
        "rollout",
        "status",
        "deployment/model-serving",
        "-n",
        "mlops-kubeflow-pipeline",
        "--timeout=60s",
    ]


def test_wait_for_rollout_reports_failure_on_timeout_or_stuck_rollout():
    with patch(
        "subprocess.run",
        return_value=_completed_process(
            1, stderr="error: timed out waiting for the condition\n"
        ),
    ):
        result = wait_for_rollout("model-serving", "mlops-kubeflow-pipeline", timeout_seconds=5)

    assert result.success is False
    assert "timed out" in result.output
