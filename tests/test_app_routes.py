"""
Tests for the FastAPI serving layer's routes, dependency injection,
and startup lifecycle.

Uses isolated fixture artifacts (a tiny trained model + hand-written
training_report.json/metrics.json) rather than the real pipeline's
output, so these tests never depend on whatever a real pipeline run
happened to produce most recently.
"""

import json

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

import app.contract as contract_module
import app.model_loader as model_loader_module
import app.main as main_module
from app.main import create_app
from app.model_loader import ModelBundle, get_model_bundle


def _write_fixture_artifacts(tmp_path):
    X_train = pd.DataFrame(
        {"feature_a": [1.0, 2.0, 3.0, 4.0], "feature_b": [4, 3, 2, 1]}
    )
    y_train = pd.Series(["a", "b", "a", "b"], name="species")

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    model_dir = tmp_path / "model"
    artifact_dir = tmp_path / "artifacts"
    model_dir.mkdir()
    artifact_dir.mkdir()

    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)

    training_report = {
        "algorithm": "RandomForestClassifier",
        "framework": "sklearn",
        "model_version": "v1.0.0",
        "trained_at": "2026-01-01T00:00:00+00:00",
        "training_samples": 4,
        "num_features": 2,
        "model_path": str(model_path),
        "feature_names": ["feature_a", "feature_b"],
        "feature_dtypes": {"feature_a": "float64", "feature_b": "int64"},
        "target_column": "species",
        "class_labels": ["a", "b"],
    }
    (artifact_dir / "training_report.json").write_text(json.dumps(training_report))

    metrics = {
        "dataset": "test_dataset",
        "algorithm": "RandomForestClassifier",
        "model_version": "v1.0.0",
        "trained_at": "2026-01-01T00:00:00+00:00",
        "evaluated_at": "2026-01-01T00:05:00+00:00",
        "train": {
            "num_samples": 4,
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
        },
        "test": {
            "num_samples": 2,
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "confusion_matrix": {"labels": ["a", "b"], "matrix": [[1, 0], [0, 1]]},
        },
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics))

    return model_dir, artifact_dir


def _build_test_app(tmp_path, monkeypatch):
    model_dir, artifact_dir = _write_fixture_artifacts(tmp_path)

    monkeypatch.setattr(contract_module, "ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(model_loader_module, "ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(model_loader_module, "MODEL_DIR", model_dir)

    return create_app()


def test_health_endpoint_returns_ok(tmp_path, monkeypatch):
    app = _build_test_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metadata_endpoint_reflects_training_contract(tmp_path, monkeypatch):
    app = _build_test_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        response = client.get("/v1/metadata")

    body = response.json()

    assert response.status_code == 200
    assert body["algorithm"] == "RandomForestClassifier"
    assert body["feature_names"] == ["feature_a", "feature_b"]
    assert body["class_labels"] == ["a", "b"]
    assert "model_path" not in body


def test_metrics_endpoint_reflects_metrics_json(tmp_path, monkeypatch):
    app = _build_test_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        response = client.get("/v1/metrics")

    body = response.json()

    assert response.status_code == 200
    assert body["dataset"] == "test_dataset"
    assert body["test"]["confusion_matrix"]["labels"] == ["a", "b"]


def test_predict_returns_valid_prediction(tmp_path, monkeypatch):
    app = _build_test_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/v1/predict", json={"feature_a": 1.0, "feature_b": 4}
        )

    assert response.status_code == 200
    assert response.json()["prediction"] in ["a", "b"]


def test_predict_batch_returns_one_prediction_per_record(tmp_path, monkeypatch):
    app = _build_test_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/v1/predict/batch",
            json={
                "records": [
                    {"feature_a": 1.0, "feature_b": 4},
                    {"feature_a": 4.0, "feature_b": 1},
                ]
            },
        )

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_predict_rejects_missing_required_field(tmp_path, monkeypatch):
    app = _build_test_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        response = client.post("/v1/predict", json={"feature_a": 1.0})

    assert response.status_code == 422


def test_predict_rejects_invalid_datatype(tmp_path, monkeypatch):
    app = _build_test_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/v1/predict", json={"feature_a": "not_a_number", "feature_b": 4}
        )

    assert response.status_code == 422


def test_predict_rejects_unknown_field(tmp_path, monkeypatch):
    app = _build_test_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/v1/predict",
            json={"feature_a": 1.0, "feature_b": 4, "unexpected_field": 123},
        )

    assert response.status_code == 422


def test_openapi_schema_reflects_dynamic_feature_contract(tmp_path, monkeypatch):
    app = _build_test_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    prediction_schema = schema["components"]["schemas"]["PredictionRequest"]

    assert set(prediction_schema["required"]) == {"feature_a", "feature_b"}
    assert prediction_schema["properties"]["feature_a"]["type"] == "number"
    assert prediction_schema["properties"]["feature_b"]["type"] == "integer"


def test_predict_uses_overridden_model_bundle_via_dependency_injection(
    tmp_path, monkeypatch
):
    app = _build_test_app(tmp_path, monkeypatch)

    class StubModel:
        def predict(self, df):
            return ["stub_prediction"] * len(df)

    stub_bundle = ModelBundle(
        model=StubModel(),
        contract={
            "feature_names": ["feature_a", "feature_b"],
            "feature_dtypes": {"feature_a": "float64", "feature_b": "int64"},
        },
        metrics={},
    )

    app.dependency_overrides[get_model_bundle] = lambda: stub_bundle

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/predict", json={"feature_a": 1.0, "feature_b": 4}
            )
    finally:
        app.dependency_overrides.clear()

    assert response.json()["prediction"] == "stub_prediction"


def test_model_bundle_loaded_exactly_once_at_startup(tmp_path, monkeypatch):
    model_dir, artifact_dir = _write_fixture_artifacts(tmp_path)

    monkeypatch.setattr(contract_module, "ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(model_loader_module, "ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(model_loader_module, "MODEL_DIR", model_dir)

    call_count = {"count": 0}
    original_load_model_bundle = model_loader_module.load_model_bundle

    def counting_load_model_bundle(contract):
        call_count["count"] += 1
        return original_load_model_bundle(contract)

    monkeypatch.setattr(main_module, "load_model_bundle", counting_load_model_bundle)

    app = create_app()

    with TestClient(app) as client:
        client.get("/v1/health")
        client.get("/v1/metadata")
        client.get("/v1/health")

    assert call_count["count"] == 1
