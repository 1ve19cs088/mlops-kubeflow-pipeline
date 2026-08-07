"""
Placeholder registry for model-lifecycle actions (promote, archive,
rollback, deploy).

None of these are implemented yet. Every entry here only describes
what a future stage would do — this module performs no MLflow stage
transition, no Docker build, no Kubernetes call, nothing. Keeping the
descriptions here (rather than in dashboard/routes.py) means the
route layer never has to know what "deploying a model" actually
entails; it only renders whichever description this module returns.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DeploymentAction:
    key: str
    label: str
    description: str
    future_stage: str


ACTIONS = {
    "promote": DeploymentAction(
        key="promote",
        label="Promote to Production",
        description=(
            "Would transition this model version's MLflow registry stage "
            'to "Production", making it the version that later stages '
            "(rollback, deployment) would treat as the active one."
        ),
        future_stage="A later Rollback phase in this project's roadmap.",
    ),
    "archive": DeploymentAction(
        key="archive",
        label="Archive Version",
        description=(
            "Would transition this model version's MLflow registry stage "
            'to "Archived", marking it retired without deleting its run '
            "history or logged artifacts."
        ),
        future_stage="A later Rollback phase in this project's roadmap.",
    ),
    "rollback": DeploymentAction(
        key="rollback",
        label="Rollback to this Version",
        description=(
            'Would re-promote this specific version to "Production", '
            "reverting whichever version is currently serving traffic "
            "back to this one."
        ),
        future_stage="A dedicated Rollback phase in this project's roadmap.",
    ),
    "deploy": DeploymentAction(
        key="deploy",
        label="Deploy",
        description=(
            "Would build a container image for this version, push it to "
            "a container registry, and roll out a Kubernetes deployment "
            "update so it actually serves traffic."
        ),
        future_stage=(
            "The Container Registry and CD phases later in this "
            "project's roadmap."
        ),
    ),
}


def get_action(key: str) -> Optional[DeploymentAction]:
    return ACTIONS.get(key)
