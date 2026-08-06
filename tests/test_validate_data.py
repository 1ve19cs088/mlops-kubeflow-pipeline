"""
Tests for the data validation stage.
"""

import pandas as pd
import pytest

import src.data.validate_data as validate_data_module
from src.data.validate_data import (
    DataValidationError,
    ValidationArtifacts,
    run,
    validate_schema,
)

CONFIG = {
    "schema": {
        "target_column": "species",
        "expected_columns": ["sepal_length", "species"],
        "valid_labels": ["setosa", "versicolor"],
    }
}


def test_validate_schema_passes_for_valid_data():
    df = pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9],
            "species": ["setosa", "versicolor"],
        }
    )

    report = validate_schema(df, CONFIG)

    assert report["valid"] is True
    assert report["missing_columns"] == []
    assert report["unexpected_labels"] == []


def test_validate_schema_flags_missing_column():
    df = pd.DataFrame({"species": ["setosa"]})

    report = validate_schema(df, CONFIG)

    assert report["valid"] is False
    assert "sepal_length" in report["missing_columns"]


def test_validate_schema_flags_missing_values():
    df = pd.DataFrame(
        {
            "sepal_length": [5.1, None],
            "species": ["setosa", "versicolor"],
        }
    )

    report = validate_schema(df, CONFIG)

    assert report["valid"] is False
    assert report["missing_values"]["sepal_length"] == 1


def test_validate_schema_flags_unexpected_label():
    df = pd.DataFrame(
        {
            "sepal_length": [5.1],
            "species": ["unknown_species"],
        }
    )

    report = validate_schema(df, CONFIG)

    assert report["valid"] is False
    assert "unknown_species" in report["unexpected_labels"]


def test_run_raises_on_invalid_data(tmp_path, monkeypatch):
    monkeypatch.setattr(validate_data_module, "ARTIFACT_DIR", tmp_path)

    df = pd.DataFrame({"species": ["setosa"]})

    with pytest.raises(DataValidationError):
        run(df, CONFIG)


def test_run_returns_validation_artifacts_on_valid_data(tmp_path, monkeypatch):
    monkeypatch.setattr(validate_data_module, "ARTIFACT_DIR", tmp_path)

    df = pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9],
            "species": ["setosa", "versicolor"],
        }
    )

    result = run(df, CONFIG)

    assert isinstance(result, ValidationArtifacts)
    pd.testing.assert_frame_equal(result.dataframe, df)
    assert result.report_path.exists()
    assert result.report_path == tmp_path / "validation_report.json"
