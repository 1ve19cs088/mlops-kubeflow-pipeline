"""
Tests for dynamic schema construction in the FastAPI serving layer.

These are pure unit tests of business logic — no FastAPI app, no
HTTP, no TestClient. They verify the contract-to-Pydantic-model
translation in isolation.
"""

import pytest
from pydantic import ValidationError

from app.schemas import (
    _python_type_for,
    build_batch_request_model,
    build_prediction_request_model,
)

CONTRACT = {
    "feature_names": ["feature_a", "feature_b"],
    "feature_dtypes": {"feature_a": "float64", "feature_b": "int64"},
}


def test_python_type_for_known_dtypes():
    assert _python_type_for("float64") is float
    assert _python_type_for("int64") is int
    assert _python_type_for("bool") is bool
    assert _python_type_for("object") is str
    assert _python_type_for("str") is str
    assert _python_type_for("string") is str


def test_python_type_for_unknown_dtype_falls_back_to_str():
    assert _python_type_for("datetime64[ns]") is str


def test_build_prediction_request_model_has_expected_fields_and_types():
    model = build_prediction_request_model(CONTRACT)

    fields = model.model_fields

    assert set(fields.keys()) == {"feature_a", "feature_b"}
    assert fields["feature_a"].annotation is float
    assert fields["feature_b"].annotation is int
    assert fields["feature_a"].is_required()
    assert fields["feature_b"].is_required()


def test_build_prediction_request_model_rejects_missing_field():
    model = build_prediction_request_model(CONTRACT)

    with pytest.raises(ValidationError):
        model(feature_a=1.0)


def test_build_prediction_request_model_rejects_unknown_field():
    model = build_prediction_request_model(CONTRACT)

    with pytest.raises(ValidationError):
        model(feature_a=1.0, feature_b=2, unexpected_field=123)


def test_build_batch_request_model_wraps_list_of_prediction_model():
    prediction_model = build_prediction_request_model(CONTRACT)
    batch_model = build_batch_request_model(prediction_model)

    instance = batch_model(records=[{"feature_a": 1.0, "feature_b": 2}])

    assert len(instance.records) == 1
    assert isinstance(instance.records[0], prediction_model)


def test_build_batch_request_model_rejects_unknown_top_level_field():
    prediction_model = build_prediction_request_model(CONTRACT)
    batch_model = build_batch_request_model(prediction_model)

    with pytest.raises(ValidationError):
        batch_model(records=[{"feature_a": 1.0, "feature_b": 2}], extra_key=True)
