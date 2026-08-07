"""
Tests for deployment/registry_client.py.

Mocks httpx.get entirely — no real network call happens in this
suite, no registry credentials are used, no Docker daemon is needed.
"""

from unittest.mock import MagicMock, patch

import httpx

from deployment.config import DeploymentConfig
from deployment.registry_client import get_latest_published_image


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


def _mock_response(status_code=200, json_data=None, headers=None):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data or {}
    response.headers = headers or {}
    if status_code >= 400 and status_code != 404:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=response
        )
    else:
        response.raise_for_status.return_value = None
    return response


def test_returns_published_image_when_manifest_digest_is_found():
    token_response = _mock_response(json_data={"token": "anon-token"})
    manifest_response = _mock_response(
        headers={"Docker-Content-Digest": "sha256:abc123"}
    )

    with patch("httpx.get", side_effect=[token_response, manifest_response]) as mock_get:
        result = get_latest_published_image(_config())

    assert result is not None
    assert result.tag == "latest"
    assert result.digest == "sha256:abc123"
    assert mock_get.call_count == 2


def test_returns_none_when_manifest_is_not_found():
    token_response = _mock_response(json_data={"token": "anon-token"})
    not_found_response = _mock_response(status_code=404)

    with patch("httpx.get", side_effect=[token_response, not_found_response]):
        result = get_latest_published_image(_config())

    assert result is None


def test_returns_none_on_any_network_failure():
    with patch("httpx.get", side_effect=httpx.ConnectError("network unreachable")):
        result = get_latest_published_image(_config())

    assert result is None


def test_returns_none_for_unsupported_provider():
    result = get_latest_published_image(_config(provider="dockerhub"))

    assert result is None


def test_returns_none_when_not_configured():
    result = get_latest_published_image(_config(repository=""))

    assert result is None


def test_no_call_made_when_provider_is_not_ghcr():
    with patch("httpx.get") as mock_get:
        get_latest_published_image(_config(provider="dockerhub"))

    mock_get.assert_not_called()
