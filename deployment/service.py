"""
DeploymentService: the single point other code goes through to ask
"what would this project deploy, and where would it go."

This stage introduces the abstraction and its data shape only — no
push, deploy, or rollout behavior exists yet. No Docker or Kubernetes
client is imported here, and no shell command is ever run. Every
field comes from DeploymentConfig; nothing is hardcoded here.
"""

from dataclasses import dataclass
from typing import Optional

from deployment.config import DeploymentConfig, load_deployment_config


@dataclass(frozen=True)
class DeploymentStatus:
    registry: str
    repository: str
    image_name: str
    latest_image_tag: str
    deployment_target: str
    current_deployed_version: str
    status: str


class DeploymentService:
    def __init__(self, config: Optional[DeploymentConfig] = None):
        self._config = config or load_deployment_config()

    def get_status(self) -> DeploymentStatus:
        """
        The current deployment configuration, plus a status of
        "Configured" or "Not Configured" — computed from whether a
        registry/repository/image name are actually set, not from any
        live registry or cluster check (this stage makes none).
        """

        config = self._config
        is_configured = bool(config.registry and config.repository and config.image_name)

        return DeploymentStatus(
            registry=config.registry,
            repository=config.repository,
            image_name=config.image_name,
            latest_image_tag=config.latest_image_tag,
            deployment_target=config.deployment_target,
            current_deployed_version=config.current_deployed_version,
            status="Configured" if is_configured else "Not Configured",
        )


def get_deployment_service() -> DeploymentService:
    """FastAPI dependency provider — overridable in tests."""

    return DeploymentService()
