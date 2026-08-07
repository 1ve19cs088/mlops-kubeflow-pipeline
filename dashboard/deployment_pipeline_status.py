"""
Read-only assessment of which stages of this project's deployment
pipeline (Developer -> Git -> CI -> Train -> Register -> Build Docker
-> Deploy -> Rolling Update) actually exist today.

No Docker, Kubernetes, or shell calls happen here — only checks for
the presence of files this project already commits (the pipeline
entrypoint, the MLflow tracking module, the Dockerfiles). Two stages
are honestly fixed rather than computed, because no file or config
exists yet to check for: there's no CI job that deploys to
Kubernetes, and no mechanism tracks what's actually live in a
cluster. A fixed status there isn't fakery — it's the true state of a
stage that hasn't been built.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AVAILABLE = "Available"
NOT_CONFIGURED = "Not Configured"
FUTURE_STAGE = "Future Stage"


def get_deployment_pipeline_status(project_root: Path = PROJECT_ROOT) -> dict:
    training = (
        AVAILABLE
        if (project_root / "src" / "pipeline" / "run_pipeline.py").exists()
        else NOT_CONFIGURED
    )
    registry = (
        AVAILABLE
        if (project_root / "src" / "tracking" / "mlflow_tracking.py").exists()
        else NOT_CONFIGURED
    )
    docker_build = (
        AVAILABLE
        if (project_root / "docker" / "Dockerfile.serving").exists()
        and (project_root / "docker" / "Dockerfile.pipeline").exists()
        else NOT_CONFIGURED
    )

    return {
        "training": training,
        "registry": registry,
        "docker_build": docker_build,
        "deployment": FUTURE_STAGE,
        "current_deployment": NOT_CONFIGURED,
    }
