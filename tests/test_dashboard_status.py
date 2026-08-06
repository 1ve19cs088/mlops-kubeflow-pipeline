"""
Tests for the dashboard's System Status page.
"""

from fastapi.testclient import TestClient

from dashboard.api_client import ApiUnavailableError, get_api_client
from dashboard.main import app


class StubApiClient:
    def get_health(self):
        return {"status": "ok"}

    def get_metadata(self):
        return {"algorithm": "RandomForestClassifier"}


class UnavailableApiClient:
    def get_health(self):
        raise ApiUnavailableError("connection refused")

    def get_metadata(self):
        raise ApiUnavailableError("connection refused")


def test_status_page_shows_healthy_when_api_and_model_available():
    app.dependency_overrides[get_api_client] = lambda: StubApiClient()

    try:
        with TestClient(app) as client:
            response = client.get("/status")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "Healthy" in response.text
    assert "RandomForestClassifier" in response.text
    assert "actions/workflows/ci.yml/badge.svg" in response.text


def test_status_page_degrades_gracefully_when_api_unavailable():
    app.dependency_overrides[get_api_client] = lambda: UnavailableApiClient()

    try:
        with TestClient(app) as client:
            response = client.get("/status")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "Unreachable" in response.text
    assert "Unavailable" in response.text
