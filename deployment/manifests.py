"""
Loads this project's already-committed Kubernetes manifests
(kubernetes/*.yaml) and produces the exact multi-document YAML text a
deploy should hand to `kubectl apply`.

The only thing ever computed rather than read verbatim is the
Deployment's container image — everything else (replicas, probes,
resources, the namespace/service/hpa definitions) comes straight from
the committed files, so a deploy can never drift from what's actually
checked into the repository.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = PROJECT_ROOT / "kubernetes"

NAMESPACE_MANIFEST = MANIFESTS_DIR / "namespace.yaml"
DEPLOYMENT_MANIFEST = MANIFESTS_DIR / "deployment.yaml"
SERVICE_MANIFEST = MANIFESTS_DIR / "service.yaml"
HPA_MANIFEST = MANIFESTS_DIR / "hpa.yaml"


def get_deployment_identity() -> tuple[str, str]:
    """
    The Deployment's name and namespace, read directly from
    kubernetes/deployment.yaml — for read-only lookups (e.g. querying
    the live cluster for its current image) that don't need to patch
    an image at all.
    """

    deployment = yaml.safe_load(DEPLOYMENT_MANIFEST.read_text())
    return deployment["metadata"]["name"], deployment["metadata"]["namespace"]


def build_manifest_bundle(image: str) -> tuple[str, str, str]:
    """
    Returns (manifest_yaml, deployment_name, namespace).

    manifest_yaml is namespace + deployment (container image swapped
    to `image`) + service + hpa, joined as one multi-document YAML
    string in the order they must be created in — ready to pipe
    straight into `kubectl apply -f -`.

    deployment_name/namespace are read from kubernetes/deployment.yaml
    itself, never assumed, so this stays correct if the manifest is
    ever renamed or moved to a different namespace.
    """

    deployment = yaml.safe_load(DEPLOYMENT_MANIFEST.read_text())
    deployment["spec"]["template"]["spec"]["containers"][0]["image"] = image

    deployment_name = deployment["metadata"]["name"]
    namespace = deployment["metadata"]["namespace"]

    documents = [
        NAMESPACE_MANIFEST.read_text(),
        yaml.safe_dump(deployment),
        SERVICE_MANIFEST.read_text(),
        HPA_MANIFEST.read_text(),
    ]
    manifest_yaml = "\n---\n".join(documents)

    return manifest_yaml, deployment_name, namespace
