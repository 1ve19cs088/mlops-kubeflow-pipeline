"""
Deployment info sourced from this repo's own kubernetes/ manifests and
safe, read-only local environment markers.

Deliberately NOT a live cluster query: the project has a standing rule
that automated code must never interact with any Kubernetes cluster
except the local kind-ai-agent one, and doing that safely would need
careful context verification this dashboard has no way to guarantee.
Reading the already-committed desired-state YAML avoids that risk
entirely while still answering "what would this deploy as."
"""

import os
from pathlib import Path

import yaml

KUBERNETES_DIR = Path(__file__).resolve().parent.parent / "kubernetes"
DOCKER_ENV_MARKER = Path("/.dockerenv")
KUBERNETES_SERVICEACCOUNT_MARKER = Path("/var/run/secrets/kubernetes.io/serviceaccount")


def _load_yaml(filename: str):
    path = KUBERNETES_DIR / filename
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())


def is_running_in_docker() -> bool:
    return DOCKER_ENV_MARKER.exists()


def is_running_in_kubernetes() -> bool:
    return KUBERNETES_SERVICEACCOUNT_MARKER.exists()


def get_deployment_info() -> dict:
    namespace_manifest = _load_yaml("namespace.yaml")
    deployment_manifest = _load_yaml("deployment.yaml")

    info = {
        "environment": (
            "kubernetes"
            if is_running_in_kubernetes()
            else "docker" if is_running_in_docker() else "local"
        ),
        "docker": is_running_in_docker(),
        "kubernetes": is_running_in_kubernetes(),
        "namespace": None,
        "deployment_name": None,
        "replica_count": None,
        "container_image": None,
        "api_base_url": os.environ.get("API_BASE_URL", "http://localhost:8000"),
    }

    if namespace_manifest:
        info["namespace"] = namespace_manifest.get("metadata", {}).get("name")

    if deployment_manifest:
        info["deployment_name"] = deployment_manifest.get("metadata", {}).get("name")
        spec = deployment_manifest.get("spec", {})
        info["replica_count"] = spec.get("replicas")
        containers = (
            spec.get("template", {}).get("spec", {}).get("containers", [])
        )
        if containers:
            info["container_image"] = containers[0].get("image")

    return info
