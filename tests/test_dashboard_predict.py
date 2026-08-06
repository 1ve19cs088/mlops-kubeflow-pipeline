"""
Tests for the dashboard's Prediction page.

Uses a 6-feature, non-Iris metadata stub specifically to prove the
form is genuinely generated from feature_names/feature_dtypes rather
than hardcoded — a dataset swap should never require a template edit.
"""

from fastapi.testclient import TestClient

from dashboard.api_client import (
    ApiUnavailableError,
    ApiValidationError,
    get_api_client,
)
from dashboard.main import app

SIX_FEATURE_METADATA = {
    "algorithm": "RandomForestClassifier",
    "framework": "sklearn",
    "model_version": "v2.0.0",
    "trained_at": "2026-01-01T00:00:00+00:00",
    "feature_names": ["f1", "f2", "f3", "f4", "f5", "f6"],
    "feature_dtypes": {
        "f1": "float64",
        "f2": "float64",
        "f3": "int64",
        "f4": "int64",
        "f5": "bool",
        "f6": "object",
    },
    "target_column": "target",
    "class_labels": ["yes", "no"],
}

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


class StubApiClient:
    def __init__(self, metadata, predict_result=None, predict_error=None):
        self._metadata = metadata
        self._predict_result = predict_result
        self._predict_error = predict_error
        self.last_payload = None

    def get_metadata(self):
        return self._metadata

    def predict(self, payload):
        self.last_payload = payload
        if self._predict_error:
            raise self._predict_error
        return self._predict_result


class UnavailableApiClient:
    def get_metadata(self):
        raise ApiUnavailableError("connection refused")


def test_predict_form_renders_one_input_per_feature_for_six_feature_model():
    stub = StubApiClient(SIX_FEATURE_METADATA)
    app.dependency_overrides[get_api_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/predict")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    for feature in SIX_FEATURE_METADATA["feature_names"]:
        assert f'name="{feature}"' in response.text


def test_predict_form_renders_iris_inputs_unchanged():
    stub = StubApiClient(IRIS_METADATA)
    app.dependency_overrides[get_api_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.get("/predict")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert 'name="sepal_length_cm"' in response.text
    assert 'name="petal_width_cm"' in response.text


def test_predict_submit_coerces_types_and_calls_api():
    stub = StubApiClient(IRIS_METADATA, predict_result={"prediction": "setosa"})
    app.dependency_overrides[get_api_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.post(
                "/predict",
                data={
                    "sepal_length_cm": "5.1",
                    "sepal_width_cm": "3.5",
                    "petal_length_cm": "1.4",
                    "petal_width_cm": "0.2",
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "setosa" in response.text
    assert "not available from current API" in response.text
    assert stub.last_payload == {
        "sepal_length_cm": 5.1,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
        "petal_width_cm": 0.2,
    }
    assert all(isinstance(v, float) for v in stub.last_payload.values())


def test_predict_submit_shows_api_validation_error():
    stub = StubApiClient(
        IRIS_METADATA, predict_error=ApiValidationError(422, "missing field")
    )
    app.dependency_overrides[get_api_client] = lambda: stub

    try:
        with TestClient(app) as client:
            response = client.post(
                "/predict",
                data={
                    "sepal_length_cm": "5.1",
                    "sepal_width_cm": "3.5",
                    "petal_length_cm": "1.4",
                    "petal_width_cm": "0.2",
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "Invalid input" in response.text


def test_predict_page_degrades_gracefully_when_api_unavailable():
    app.dependency_overrides[get_api_client] = lambda: UnavailableApiClient()

    try:
        with TestClient(app) as client:
            response = client.get("/predict")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "API unreachable" in response.text
