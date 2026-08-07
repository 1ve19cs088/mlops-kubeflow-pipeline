"""
MLflow client for the dashboard.

Encapsulates ALL MLflow interaction — nothing else in dashboard/
should import mlflow directly. This keeps the dashboard's dependency
on MLflow's specific API surface contained to one module, the same
way ApiClient contains its dependency on the serving API's HTTP
surface.

Computes its own default tracking URI rather than importing
src.config.settings.PROJECT_ROOT, to preserve the existing rule that
dashboard/ never imports app/ or src/ — reading the same MLflow store
the pipeline writes to is a data-access integration, not a business-
logic dependency, so this only needs to agree on *where* that store
is, not on any of the pipeline's code.
"""

import os
from pathlib import Path
from typing import Optional

import mlflow.artifacts
from mlflow import MlflowClient
from mlflow.entities import FileInfo
from mlflow.entities.model_registry import ModelVersion, RegisteredModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT}/mlflow.db"
)

PRODUCTION_STAGE = "Production"


class MlflowRegistryClient:
    """
    Thin wrapper around mlflow.MlflowClient, scoped to one tracking
    URI. Every dashboard page that needs registry data goes through
    this class — never mlflow directly.

    Uses MlflowClient(tracking_uri=...) rather than the global
    mlflow.set_tracking_uri()/mlflow.<function> style used in
    src/tracking/mlflow_tracking.py: this avoids mutating any global
    state at all, which matters here because a single dashboard
    process may serve many requests concurrently — there's no shared
    "current tracking URI" to race on.
    """

    def __init__(self, tracking_uri: str = MLFLOW_TRACKING_URI):
        self._tracking_uri = tracking_uri
        self._client = MlflowClient(tracking_uri=tracking_uri)

    def get_registered_models(self) -> list[RegisteredModel]:
        """All registered models, most recently updated first."""

        models = list(self._client.search_registered_models())
        return sorted(models, key=lambda m: m.last_updated_timestamp, reverse=True)

    def get_model_versions(self, name: str) -> list[ModelVersion]:
        """All versions of `name`, newest version number first."""

        versions = self._client.search_model_versions(f"name='{name}'")
        return sorted(versions, key=lambda v: int(v.version), reverse=True)

    def get_latest_version(self, name: str) -> Optional[ModelVersion]:
        """The highest version number registered for `name`, if any."""

        versions = self.get_model_versions(name)
        return versions[0] if versions else None

    def get_production_version(self, name: str) -> Optional[ModelVersion]:
        """
        The version of `name` currently in the "Production" stage, if
        one has been promoted.

        Every version defaults to stage "None" (the literal string
        MLflow uses, not Python None) until explicitly transitioned —
        promotion isn't built until Phase 3, so this returns None for
        every model today. Confirmed against this project's real
        registered models before writing this, not assumed.
        """

        for version in self.get_model_versions(name):
            if version.current_stage == PRODUCTION_STAGE:
                return version
        return None

    def get_run_metrics(self, run_id: str) -> dict:
        """All metrics logged against `run_id`."""

        return dict(self._client.get_run(run_id).data.metrics)

    def get_run_parameters(self, run_id: str) -> dict:
        """All parameters logged against `run_id`."""

        return dict(self._client.get_run(run_id).data.params)

    def list_run_artifacts(self, run_id: str) -> list[FileInfo]:
        """Top-level artifact files/directories logged against `run_id`."""

        return list(self._client.list_artifacts(run_id))

    def get_artifact_bytes(self, run_id: str, artifact_path: str) -> bytes:
        """
        Downloads one artifact file and returns its raw bytes.

        Uses mlflow.artifacts.download_artifacts (scoped to this
        client's own tracking_uri, not any process-global one, for the
        same concurrency-safety reason described on this class) rather
        than MlflowClient.download_artifacts, which is deprecated.
        Downloads land in a fresh tempfile.mkdtemp() directory, not the
        project root.
        """

        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            tracking_uri=self._tracking_uri,
        )
        return Path(local_path).read_bytes()


def get_mlflow_client() -> MlflowRegistryClient:
    """FastAPI dependency provider — overridable in tests."""

    return MlflowRegistryClient()
