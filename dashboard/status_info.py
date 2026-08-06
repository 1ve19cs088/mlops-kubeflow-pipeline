"""
Non-API-derived signals for the System Status page.

API/model status come directly from ApiClient in the route (same
dependency-injection pattern as every other page); this module only
covers environment detection (reused from deployment_info) and the
GitHub Actions badge URL.
"""

import os

from dashboard.deployment_info import is_running_in_docker, is_running_in_kubernetes

GITHUB_REPO = os.environ.get("GITHUB_REPO", "1ve19cs088/mlops-kubeflow-pipeline")


def get_github_actions_badge_url() -> str:
    return f"https://github.com/{GITHUB_REPO}/actions/workflows/ci.yml/badge.svg"


def get_environment_status() -> dict:
    return {
        "docker": is_running_in_docker(),
        "kubernetes": is_running_in_kubernetes(),
    }
