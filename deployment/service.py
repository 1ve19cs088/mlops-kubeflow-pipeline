"""
DeploymentService: the single point other code goes through to ask
"what would this project deploy, and where would it go", "what's
actually published right now", and — as of this stage — to actually
deploy a published image to the cluster (with an automatic rollback
safety net if that deploy fails).

Configuration fields come from DeploymentConfig; publish-status comes
from a real, anonymous registry query; deploy() applies this
project's own committed Kubernetes manifests via kubectl and waits
for the rollout. Nothing here is hardcoded or faked, and no
shell/Docker/Kubernetes call happens anywhere except inside
deploy() — every other method stays read-only.
"""

import time
from dataclasses import dataclass
from typing import Optional

from deployment.config import DeploymentConfig, load_deployment_config
from deployment.kubectl_client import apply_manifest, rollout_undo, wait_for_rollout
from deployment.manifests import build_manifest_bundle
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


@dataclass(frozen=True)
class DeploymentResult:
    success: bool
    image: str
    namespace: str
    deployment_name: str
    duration_seconds: float
    message: str
    # Populated only when a rollout failure triggered an automatic
    # rollback (deploy() below) — None/False otherwise.
    rolled_back: bool = False
    rollback_success: Optional[bool] = None
    rollback_duration_seconds: Optional[float] = None
    rollback_message: Optional[str] = None
    original_error: Optional[str] = None


class DeploymentService:
    def __init__(self, config: Optional[DeploymentConfig] = None):
        self._config = config or load_deployment_config()

    def _build_image_reference(self, tag: str) -> str:
        config = self._config
        owner = config.repository.split("/")[0]
        return f"{config.registry}/{owner}/{config.image_name}:{tag}"

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
            latest_published_image = self._build_image_reference(published_image.tag)

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

    def deploy(self, image_tag: str, timeout_seconds: int = 120) -> DeploymentResult:
        """
        Points the model-serving Deployment at `image_tag`, applies
        this project's own committed Kubernetes manifests (namespace,
        deployment, service, hpa — kubernetes/*.yaml, unmodified
        except for the image field) via kubectl, and waits for the
        rollout to finish.

        If the rollout fails, the Deployment's desired image would
        otherwise be left pointing at the broken image indefinitely —
        unsafe, since a node restart, pod eviction, or future rollout
        could all try to use it again. So a failed rollout triggers an
        automatic `kubectl rollout undo` (Kubernetes' own revision
        history, not anything tracked here) followed by a second
        rollout wait to confirm the revert itself actually completes.

        This is the first method on DeploymentService that changes
        cluster state — get_status() and everything before it in this
        project stayed read-only. Every failure mode (apply rejected,
        rollout failed, rollback command rejected, rollback rollout
        failed) is reported in the result rather than raised; nothing
        here assumes the cluster or kubectl are reachable.
        """

        image = self._build_image_reference(image_tag)
        manifest_yaml, deployment_name, namespace = build_manifest_bundle(image)

        start = time.monotonic()

        apply_result = apply_manifest(manifest_yaml)
        if not apply_result.success:
            # Nothing to roll back: the Deployment's desired state was
            # never actually changed if kubectl rejected the manifest.
            message = f"kubectl apply failed: {apply_result.output}"
            return DeploymentResult(
                success=False,
                image=image,
                namespace=namespace,
                deployment_name=deployment_name,
                duration_seconds=time.monotonic() - start,
                message=message,
                original_error=message,
            )

        rollout_result = wait_for_rollout(deployment_name, namespace, timeout_seconds)
        duration_seconds = time.monotonic() - start

        if rollout_result.success:
            return DeploymentResult(
                success=True,
                image=image,
                namespace=namespace,
                deployment_name=deployment_name,
                duration_seconds=duration_seconds,
                message=rollout_result.output or "Rollout completed successfully.",
            )

        original_error = f"Rollout failed or timed out: {rollout_result.output}"

        rollback_start = time.monotonic()
        undo_result = rollout_undo(deployment_name, namespace)

        if not undo_result.success:
            return DeploymentResult(
                success=False,
                image=image,
                namespace=namespace,
                deployment_name=deployment_name,
                duration_seconds=duration_seconds,
                message=original_error,
                rolled_back=False,
                rollback_success=False,
                rollback_duration_seconds=time.monotonic() - rollback_start,
                rollback_message=f"kubectl rollout undo failed: {undo_result.output}",
                original_error=original_error,
            )

        rollback_rollout_result = wait_for_rollout(deployment_name, namespace, timeout_seconds)
        rollback_duration_seconds = time.monotonic() - rollback_start

        if rollback_rollout_result.success:
            return DeploymentResult(
                success=False,
                image=image,
                namespace=namespace,
                deployment_name=deployment_name,
                duration_seconds=duration_seconds,
                message=original_error,
                rolled_back=True,
                rollback_success=True,
                rollback_duration_seconds=rollback_duration_seconds,
                rollback_message=(
                    rollback_rollout_result.output or "Rollback completed successfully."
                ),
                original_error=original_error,
            )

        return DeploymentResult(
            success=False,
            image=image,
            namespace=namespace,
            deployment_name=deployment_name,
            duration_seconds=duration_seconds,
            message=original_error,
            rolled_back=False,
            rollback_success=False,
            rollback_duration_seconds=rollback_duration_seconds,
            rollback_message=(
                f"Rollback rollout failed or timed out: {rollback_rollout_result.output}"
            ),
            original_error=original_error,
        )


def get_deployment_service() -> DeploymentService:
    """FastAPI dependency provider — overridable in tests."""

    return DeploymentService()
