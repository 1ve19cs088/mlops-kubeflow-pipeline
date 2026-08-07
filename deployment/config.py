"""
Deployment configuration.

Everything here is read from environment variables with sensible
defaults, following this project's existing configuration pattern
(API_BASE_URL, MLFLOW_TRACKING_URI, GITHUB_REPO, ...). This module
holds no credentials or secrets, and performs no registry, Docker, or
Kubernetes call of any kind — it only describes *where* a deployable
image would live, not how to authenticate to push or pull one.
"""

import os
from dataclasses import dataclass

REGISTRY_HOSTS = {
    "ghcr": "ghcr.io",
    "dockerhub": "docker.io",
}


@dataclass(frozen=True)
class DeploymentConfig:
    provider: str
    registry: str
    repository: str
    image_name: str
    latest_image_tag: str
    deployment_target: str
    current_deployed_version: str


def load_deployment_config() -> DeploymentConfig:
    provider = os.environ.get("DEPLOYMENT_REGISTRY_PROVIDER", "ghcr").strip().lower()
    default_registry = REGISTRY_HOSTS.get(provider, "")

    return DeploymentConfig(
        provider=provider,
        registry=os.environ.get("DEPLOYMENT_REGISTRY", default_registry),
        # Same GitHub org/repo this project's CI badge already points
        # at (dashboard/status_info.py's GITHUB_REPO) — the natural
        # GHCR repository path if these images were ever pushed.
        repository=os.environ.get(
            "DEPLOYMENT_REPOSITORY", "1ve19cs088/mlops-kubeflow-pipeline"
        ),
        # Matches the serving image name CI already builds
        # (.github/workflows/ci.yml's "Build serving image" step).
        image_name=os.environ.get(
            "DEPLOYMENT_IMAGE_NAME", "mlops-kubeflow-pipeline-serving"
        ),
        # CI builds this image with push: false — nothing is actually
        # pushed to any registry yet, so there is no real tag to read.
        latest_image_tag=os.environ.get("DEPLOYMENT_LATEST_IMAGE_TAG", "not-yet-pushed"),
        # The only Kubernetes cluster this project is ever allowed to
        # touch.
        deployment_target=os.environ.get("DEPLOYMENT_TARGET", "kind-ai-agent"),
        current_deployed_version=os.environ.get(
            "DEPLOYMENT_CURRENT_VERSION", "Not Deployed"
        ),
    )
