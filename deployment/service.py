"""
DeploymentService: the single point other code goes through to ask
"what would this project deploy, and where would it go", "what's
actually published right now", "what's actually running right now",
and to actually deploy a published image to the cluster (with an
automatic rollback safety net if that deploy fails).

Configuration fields come from DeploymentConfig; publish-status and
current-cluster-state come from real, live queries (an anonymous
registry lookup, a `kubectl get`); deploy() applies this project's own
committed Kubernetes manifests via kubectl and waits for the rollout.
Nothing here is hardcoded or faked, and no shell/Docker/Kubernetes
call happens anywhere except inside get_current_deployment() and
deploy() — every other method stays read-only.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from deployment.config import DeploymentConfig, load_deployment_config
from deployment.kubectl_client import (
    apply_manifest,
    get_deployment_image,
    rollout_undo,
    wait_for_rollout,
)
from deployment.manifests import build_manifest_bundle, get_deployment_identity
from deployment.registry_client import get_latest_published_image, get_published_image


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
class DeployabilityCheck:
    deployable: bool
    image: Optional[str]
    reason: Optional[str]


@dataclass(frozen=True)
class CurrentDeployment:
    """
    What the live Deployment is actually running right now, read
    directly from the cluster. `tag` is whatever's after the last ":"
    in `image` — for anything deployed through this project's own
    deploy(), that's a git commit SHA, but this dataclass has no
    opinion about that; it's just string-splitting the image
    reference. Resolving `tag` back to an MLflow model version (if
    any) is the caller's job, via MlflowRegistryClient — this class
    stays exactly as MLflow-agnostic as the rest of this module.
    """

    image: Optional[str]
    tag: Optional[str]


@dataclass(frozen=True)
class DeploymentHistoryEntry:
    timestamp: str
    image: str
    tag: str
    success: bool
    rolled_back: bool
    duration_seconds: float


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
        # This process's deploy() calls so far — never persisted,
        # cleared the moment the dashboard restarts. Instance-level
        # (not module-level) so tests constructing their own
        # DeploymentService never share or pollute this list; the
        # single shared instance FastAPI's dependency injection hands
        # out (get_deployment_service(), below) is what makes this
        # accumulate across real dashboard requests within one run.
        self._history: list[DeploymentHistoryEntry] = []

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

    def get_current_deployment(self) -> CurrentDeployment:
        """
        A real, live `kubectl get` read of what the model-serving
        Deployment is actually running right now — never config,
        never a cache. Returns CurrentDeployment(None, None) if the
        cluster or Deployment can't be reached, same as every other
        "nothing to report" case in this module.
        """

        deployment_name, namespace = get_deployment_identity()
        image = get_deployment_image(deployment_name, namespace)

        tag = None
        if image and ":" in image:
            tag = image.rsplit(":", 1)[-1]

        return CurrentDeployment(image=image, tag=tag)

    def get_deployment_history(self) -> list:
        """
        This process's deploy() calls so far, newest first. Empty
        until the first deploy() call in this run — nothing is ever
        read from disk here.
        """

        return list(reversed(self._history))

    def _record(self, result: DeploymentResult) -> DeploymentResult:
        tag = result.image.rsplit(":", 1)[-1] if ":" in result.image else result.image
        self._history.append(
            DeploymentHistoryEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                image=result.image,
                tag=tag,
                success=result.success,
                rolled_back=result.rolled_back,
                duration_seconds=result.duration_seconds,
            )
        )
        return result

    def resolve_deployment_for_commit(self, git_commit_sha: Optional[str]) -> DeployabilityCheck:
        """
        Given a git commit SHA (or None), determines whether there's
        a real, published image for it — never a guess, never a
        silent fall-back to "latest".

        This method never imports or knows about MLflow: the caller
        (a future dashboard route) is responsible for resolving a
        model version to its git commit tag first, via
        MlflowRegistryClient.get_git_commit(), and passing the result
        in here. That keeps MLflow interaction isolated to
        dashboard/mlflow_client.py exactly as this project's
        architecture already requires.

        Returns DeployabilityCheck(deployable=False, image=None,
        reason=...) — with a human-readable, honest reason — for both
        ways this can fail to resolve: no commit was ever recorded for
        this version (older models, trained before commit tracking
        existed), or a commit was recorded but CI hasn't published an
        image for it (yet, or never will if that commit didn't reach
        a branch this project's CI publishes from).
        """

        if not git_commit_sha:
            return DeployabilityCheck(
                deployable=False,
                image=None,
                reason=(
                    "This model version has no recorded git commit — it was "
                    "likely trained before commit tracking was added, or "
                    "outside a git checkout. No deployable image available."
                ),
            )

        published_image = get_published_image(self._config, git_commit_sha)
        if published_image is None:
            return DeployabilityCheck(
                deployable=False,
                image=None,
                reason=(
                    f"No published image was found for commit {git_commit_sha}. "
                    "It may not have been built and pushed yet, or was never "
                    "published from a branch this project's CI publishes "
                    "images from. No deployable image available."
                ),
            )

        return DeployabilityCheck(
            deployable=True,
            image=self._build_image_reference(published_image.tag),
            reason=None,
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
            return self._record(DeploymentResult(
                success=False,
                image=image,
                namespace=namespace,
                deployment_name=deployment_name,
                duration_seconds=time.monotonic() - start,
                message=message,
                original_error=message,
            ))

        rollout_result = wait_for_rollout(deployment_name, namespace, timeout_seconds)
        duration_seconds = time.monotonic() - start

        if rollout_result.success:
            return self._record(DeploymentResult(
                success=True,
                image=image,
                namespace=namespace,
                deployment_name=deployment_name,
                duration_seconds=duration_seconds,
                message=rollout_result.output or "Rollout completed successfully.",
            ))

        original_error = f"Rollout failed or timed out: {rollout_result.output}"

        rollback_start = time.monotonic()
        undo_result = rollout_undo(deployment_name, namespace)

        if not undo_result.success:
            return self._record(DeploymentResult(
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
            ))

        rollback_rollout_result = wait_for_rollout(deployment_name, namespace, timeout_seconds)
        rollback_duration_seconds = time.monotonic() - rollback_start

        if rollback_rollout_result.success:
            return self._record(DeploymentResult(
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
            ))

        return self._record(DeploymentResult(
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
        ))


_shared_deployment_service = DeploymentService()


def get_deployment_service() -> DeploymentService:
    """
    FastAPI dependency provider — overridable in tests.

    Returns one shared instance for the life of the dashboard process
    (not a fresh DeploymentService per request), so deployment history
    actually accumulates across requests within a running dashboard —
    "current session only" per this stage's requirement, gone the
    moment the process restarts. Tests that construct their own
    DeploymentService(config=...) directly are unaffected; only code
    going through this provider shares state.
    """

    return _shared_deployment_service
