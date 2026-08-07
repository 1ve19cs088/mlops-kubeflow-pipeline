"""
DeploymentService: the single point other code goes through to ask
"what would this project deploy, and where would it go" — and, as of
this stage, "what's actually published right now."

No push, deploy, or rollout behavior exists yet. No Docker or
Kubernetes client is imported here, and no shell command is ever run.
Configuration fields come from DeploymentConfig; the
publish-status fields come from a real (but anonymous, read-only)
registry query — nothing here is hardcoded or faked.
"""

from dataclasses import dataclass
from typing import Optional

from deployment.config import DeploymentConfig, load_deployment_config
from deployment.registry_client import get_latest_published_image


@dataclass(frozen=True)
class DeploymentStatus:
    registry: str
    repository: str
    image_name: str
    latest_image_tag: str
    deployment_target: str
    current_deployed_version: str
    status: str
    latest_published_image: Optional[str]
    image_digest: Optional[str]
    current_tag: Optional[str]


class DeploymentService:
    def __init__(self, config: Optional[DeploymentConfig] = None):
        self._config = config or load_deployment_config()

    def get_status(self) -> DeploymentStatus:
        """
        The current deployment configuration, plus:
        - "status": "Configured"/"Not Configured", from whether a
          registry/repository/image name are actually set — not a
          live check.
        - "latest_published_image"/"image_digest"/"current_tag": from
          a real, anonymous query against the registry's public API
          (deployment.registry_client). None when nothing has been
          published yet, the provider isn't queryable yet (anything
          but GHCR today), or the registry can't be reached — all
          three are reported as "nothing published" identically.
        """

        config = self._config
        is_configured = bool(config.registry and config.repository and config.image_name)

        published_image = get_latest_published_image(config) if is_configured else None

        latest_published_image = None
        if published_image is not None:
            owner = config.repository.split("/")[0]
            latest_published_image = (
                f"{config.registry}/{owner}/{config.image_name}:{published_image.tag}"
            )

        return DeploymentStatus(
            registry=config.registry,
            repository=config.repository,
            image_name=config.image_name,
            latest_image_tag=config.latest_image_tag,
            deployment_target=config.deployment_target,
            current_deployed_version=config.current_deployed_version,
            status="Configured" if is_configured else "Not Configured",
            latest_published_image=latest_published_image,
            image_digest=published_image.digest if published_image else None,
            current_tag=published_image.tag if published_image else None,
        )


def get_deployment_service() -> DeploymentService:
    """FastAPI dependency provider — overridable in tests."""

    return DeploymentService()
