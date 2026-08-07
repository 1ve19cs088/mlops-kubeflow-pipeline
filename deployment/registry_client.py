"""
Read-only client for asking a container registry's public API what
image is actually published right now.

Only GHCR is implemented for live querying in this stage. Docker Hub
remains a configurable *provider* (deployment/config.py already
supports selecting it), but querying it isn't built yet — an
unsupported provider is reported as "nothing found" rather than
guessed at.

No credentials are used or required: GHCR's anonymous token endpoint
is enough to read a public package's manifest digest. Any failure —
network error, the image has never been published, the package is
private, GHCR is unreachable — is treated identically: nothing is
published as far as this dashboard can tell. Nothing here ever raises
up to the caller.
"""

from dataclasses import dataclass
from typing import Optional

import httpx

from deployment.config import DeploymentConfig

REQUEST_TIMEOUT_SECONDS = 5.0
MANIFEST_ACCEPT_HEADER = "application/vnd.docker.distribution.manifest.v2+json"


@dataclass(frozen=True)
class PublishedImage:
    tag: str
    digest: Optional[str]


def get_latest_published_image(config: DeploymentConfig) -> Optional[PublishedImage]:
    if config.provider != "ghcr":
        return None
    if not (config.registry and config.repository and config.image_name):
        return None

    owner = config.repository.split("/")[0]
    if not owner:
        return None

    repository_path = f"{owner}/{config.image_name}"

    try:
        token = _get_anonymous_pull_token(config.registry, repository_path)
        digest = _get_manifest_digest(config.registry, repository_path, "latest", token)
    except Exception:
        return None

    if digest is None:
        return None

    return PublishedImage(tag="latest", digest=digest)


def _get_anonymous_pull_token(registry: str, repository_path: str) -> str:
    response = httpx.get(
        f"https://{registry}/token",
        params={"service": registry, "scope": f"repository:{repository_path}:pull"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["token"]


def _get_manifest_digest(
    registry: str, repository_path: str, tag: str, token: str
) -> Optional[str]:
    response = httpx.get(
        f"https://{registry}/v2/{repository_path}/manifests/{tag}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": MANIFEST_ACCEPT_HEADER,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.headers.get("Docker-Content-Digest")
