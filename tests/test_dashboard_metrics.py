"""
Tests for the dashboard's Model Metrics page.
"""

from fastapi.testclient import TestClient

from dashboard.api_client import ApiUnavailableError, get_api_client
from dashboard.main import app

METRICS = {
    "dataset": "iris",
    "algorithm": "RandomForestClassifier",
    "model_version": "v1.0.0",
    "trained_at": "2026-01-01T00:00:00+00:00",
    "evaluated_at": "2026-01-01T00:05:00+00:00",
    "train": {
        "num_samples": 120,
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1_score": 1.0,
    },
    "test": {
        "num_samples": 30,
        "accuracy": 0.9667,
        "precision": 0.9675,
        "recall": 0.9667,
        "f1_score": 0.9666,
        "confusion_matrix": {
            "labels": ["setosa", "versicolor", "virginica"],
            "matrix": [[10, 0, 0], [0, 9, 1], [0, 0, 10]],
        },
    },
}


class StubApiClient:
    def get_metrics(self):
        return METRICS


class UnavailableApiClient:
    def get_metrics(self):
        raise ApiUnavailableError("connection refused")


def test_metrics_page_renders_metrics():
    app.dependency_overrides[get_api_client] = lambda: StubApiClient()

    try:
        with TestClient(app) as client:
            response = client.get("/metrics")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "iris" in response.text
    assert "0.9667" in response.text
    assert "setosa" in response.text
    assert "versicolor" in response.text


def test_metrics_page_degrades_gracefully_when_api_unavailable():
    app.dependency_overrides[get_api_client] = lambda: UnavailableApiClient()

    try:
        with TestClient(app) as client:
            response = client.get("/metrics")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "API unreachable" in response.text
