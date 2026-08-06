"""
Tests for the dashboard's Home page.

Overrides the ApiClient dependency with a stub — these tests never
make a real HTTP call to the serving API, consistent with how the
main API's own tests stub ModelBundle via dependency injection.
"""

from fastapi.testclient import TestClient

from dashboard.api_client import ApiUnavailableError, get_api_client
from dashboard.main import app


class StubApiClient:
    def get_health(self):
        return {"status": "ok"}

    def get_metadata(self):
        return {
            "algorithm": "RandomForestClassifier",
            "framework": "sklearn",
            "model_version": "v1.0.0",
            "trained_at": "2026-01-01T00:00:00+00:00",
        }

    def get_metrics(self):
        return {"dataset": "iris"}


class UnavailableApiClient:
    def get_health(self):
        raise ApiUnavailableError("connection refused")

    def get_metadata(self):
        raise ApiUnavailableError("connection refused")

    def get_metrics(self):
        raise ApiUnavailableError("connection refused")


def test_home_page_renders_data_from_api(monkeypatch):
    app.dependency_overrides[get_api_client] = lambda: StubApiClient()

    try:
        with TestClient(app) as client:
            response = client.get("/")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "RandomForestClassifier" in response.text
    assert "v1.0.0" in response.text
    assert "iris" in response.text


def test_home_page_degrades_gracefully_when_api_unavailable():
    app.dependency_overrides[get_api_client] = lambda: UnavailableApiClient()

    try:
        with TestClient(app) as client:
            response = client.get("/")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "API unreachable" in response.text
