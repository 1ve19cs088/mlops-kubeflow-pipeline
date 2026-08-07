"""
Tests for deployment/config.py.
"""

from deployment.config import load_deployment_config


def test_defaults_reflect_ghcr_and_this_projects_actual_repo(monkeypatch):
    for var in (
        "DEPLOYMENT_REGISTRY_PROVIDER",
        "DEPLOYMENT_REGISTRY",
        "DEPLOYMENT_REPOSITORY",
        "DEPLOYMENT_IMAGE_NAME",
        "DEPLOYMENT_LATEST_IMAGE_TAG",
        "DEPLOYMENT_TARGET",
        "DEPLOYMENT_CURRENT_VERSION",
    ):
        monkeypatch.delenv(var, raising=False)

    config = load_deployment_config()

    assert config.provider == "ghcr"
    assert config.registry == "ghcr.io"
    assert config.repository == "1ve19cs088/mlops-kubeflow-pipeline"
    assert config.image_name == "mlops-kubeflow-pipeline-serving"
    assert config.latest_image_tag == "not-yet-pushed"
    assert config.deployment_target == "kind-ai-agent"
    assert config.current_deployed_version == "Not Deployed"


def test_dockerhub_provider_resolves_to_docker_io(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_REGISTRY_PROVIDER", "dockerhub")
    monkeypatch.delenv("DEPLOYMENT_REGISTRY", raising=False)

    config = load_deployment_config()

    assert config.provider == "dockerhub"
    assert config.registry == "docker.io"


def test_unknown_provider_without_explicit_registry_is_empty(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_REGISTRY_PROVIDER", "something-else")
    monkeypatch.delenv("DEPLOYMENT_REGISTRY", raising=False)

    config = load_deployment_config()

    assert config.registry == ""


def test_explicit_env_vars_override_every_default(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_REGISTRY_PROVIDER", "dockerhub")
    monkeypatch.setenv("DEPLOYMENT_REGISTRY", "custom.registry.example.com")
    monkeypatch.setenv("DEPLOYMENT_REPOSITORY", "my-org/my-repo")
    monkeypatch.setenv("DEPLOYMENT_IMAGE_NAME", "custom-serving")
    monkeypatch.setenv("DEPLOYMENT_LATEST_IMAGE_TAG", "v42")
    monkeypatch.setenv("DEPLOYMENT_TARGET", "staging-cluster")
    monkeypatch.setenv("DEPLOYMENT_CURRENT_VERSION", "3")

    config = load_deployment_config()

    assert config.registry == "custom.registry.example.com"
    assert config.repository == "my-org/my-repo"
    assert config.image_name == "custom-serving"
    assert config.latest_image_tag == "v42"
    assert config.deployment_target == "staging-cluster"
    assert config.current_deployed_version == "3"


def test_no_credential_or_secret_env_vars_are_ever_read(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_REGISTRY_PASSWORD", "should-never-be-read")
    monkeypatch.setenv("DEPLOYMENT_REGISTRY_TOKEN", "should-never-be-read")

    config = load_deployment_config()

    for value in config.__dict__.values():
        assert "should-never-be-read" not in str(value)
