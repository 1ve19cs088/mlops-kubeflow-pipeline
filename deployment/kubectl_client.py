"""
Thin subprocess wrapper around `kubectl` — the only place in this
project that shells out to it.

Every call passes an explicit argument list (never shell=True), so
neither the manifest text nor the image reference flowing through
here can be interpreted as shell syntax. Any failure to even launch
kubectl (not installed, no cluster context, etc.) is caught and
reported the same way as a command that ran but failed — callers
never need to handle a raised exception.
"""

import subprocess
from dataclasses import dataclass

APPLY_TIMEOUT_SECONDS = 30
ROLLOUT_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class KubectlResult:
    success: bool
    output: str


def apply_manifest(manifest_yaml: str, timeout_seconds: int = APPLY_TIMEOUT_SECONDS) -> KubectlResult:
    """Runs `kubectl apply -f -`, piping `manifest_yaml` in via stdin."""

    try:
        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=manifest_yaml,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return KubectlResult(success=False, output=str(exc))

    output = (result.stdout + result.stderr).strip()
    return KubectlResult(success=result.returncode == 0, output=output)


def wait_for_rollout(
    deployment_name: str,
    namespace: str,
    timeout_seconds: int = ROLLOUT_TIMEOUT_SECONDS,
) -> KubectlResult:
    """
    Runs `kubectl rollout status`, which blocks until the rollout
    finishes, fails, or the given timeout elapses — kubectl itself
    detects a stuck/failed rollout (e.g. ImagePullBackOff,
    CrashLoopBackOff) as a non-zero exit once its timeout is hit.
    """

    try:
        result = subprocess.run(
            [
                "kubectl",
                "rollout",
                "status",
                f"deployment/{deployment_name}",
                "-n",
                namespace,
                f"--timeout={timeout_seconds}s",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 10,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return KubectlResult(success=False, output=str(exc))

    output = (result.stdout + result.stderr).strip()
    return KubectlResult(success=result.returncode == 0, output=output)
