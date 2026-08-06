"""
Tests for the dashboard's Batch Prediction page.
"""

import io

from fastapi.testclient import TestClient

from dashboard.api_client import ApiUnavailableError, get_api_client
from dashboard.main import app

IRIS_METADATA = {
    "algorithm": "RandomForestClassifier",
    "framework": "sklearn",
    "model_version": "v1.0.0",
    "trained_at": "2026-01-01T00:00:00+00:00",
    "feature_names": [
        "sepal_length_cm",
        "sepal_width_cm",
        "petal_length_cm",
        "petal_width_cm",
    ],
    "feature_dtypes": {
        "sepal_length_cm": "float64",
        "sepal_width_cm": "float64",
        "petal_length_cm": "float64",
        "petal_width_cm": "float64",
    },
    "target_column": "species",
    "class_labels": ["setosa", "versicolor", "virginica"],
}

CSV_CONTENT = (
    "sepal_length_cm,sepal_width_cm,petal_length_cm,petal_width_cm\n"
    "5.1,3.5,1.4,0.2\n"
    "6.7,3.1,4.7,1.5\n"
)


class StubApiClient:
    def __init__(self, metadata, predictions=None):
        self._metadata = metadata
        self._predictions = predictions or ["setosa", "versicolor"]
        self.last_records = None

    def get_metadata(self):
        return self._metadata

    def predict_batch(self, records):
        self.last_records = records
        return {"predictions": self._predictions}


class UnavailableApiClient:
    def get_metadata(self):
        raise ApiUnavailableError("connection refused")


def test_batch_preview_shows_predictions_for_each_row():
    stub = StubApiClient(IRIS_METADATA)
    app.dependency_overrides[get_api_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.post(
                "/batch",
                files={"file": ("data.csv", io.BytesIO(CSV_CONTENT.encode()), "text/csv")},
                data={"action": "preview"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "setosa" in response.text
    assert "versicolor" in response.text
    assert len(stub.last_records) == 2
    assert all(isinstance(v, float) for v in stub.last_records[0].values())


def test_batch_download_returns_csv_with_predictions():
    stub = StubApiClient(IRIS_METADATA)
    app.dependency_overrides[get_api_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.post(
                "/batch",
                files={"file": ("data.csv", io.BytesIO(CSV_CONTENT.encode()), "text/csv")},
                data={"action": "download"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    assert "prediction" in response.text
    assert "setosa" in response.text


def test_batch_rejects_csv_missing_required_columns():
    stub = StubApiClient(IRIS_METADATA)
    app.dependency_overrides[get_api_client] = lambda: stub

    bad_csv = "sepal_length_cm,sepal_width_cm\n5.1,3.5\n"

    try:
        with TestClient(app) as client:
            response = client.post(
                "/batch",
                files={"file": ("data.csv", io.BytesIO(bad_csv.encode()), "text/csv")},
                data={"action": "preview"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "missing required columns" in response.text.lower()


def test_batch_page_degrades_gracefully_when_api_unavailable():
    app.dependency_overrides[get_api_client] = lambda: UnavailableApiClient()

    try:
        with TestClient(app) as client:
            response = client.get("/batch")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "API unreachable" in response.text
