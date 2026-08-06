"""
Tests for the feature engineering stage.
"""

from pathlib import Path

import pandas as pd

import src.features.feature_engineering as feature_engineering_module
from src.data.validate_data import ValidationArtifacts
from src.features.feature_engineering import (
    FeatureEngineeringArtifacts,
    run,
    split_features_and_target,
)

CONFIG = {
    "schema": {"target_column": "species"},
    "training": {"test_size": 0.2, "random_state": 42},
}


def _sample_df(n=20):
    return pd.DataFrame(
        {
            "feature_a": range(n),
            "feature_b": [value * 2 for value in range(n)],
            "species": [
                "setosa" if value % 2 == 0 else "versicolor" for value in range(n)
            ],
        }
    )


def _sample_validation(n=20):
    return ValidationArtifacts(
        dataframe=_sample_df(n), report_path=Path("unused_report.json")
    )


def test_split_features_and_target_separates_columns():
    df = _sample_df()

    X, y = split_features_and_target(df, "species")

    assert list(X.columns) == ["feature_a", "feature_b"]
    assert y.name == "species"
    assert len(X) == len(y) == len(df)


def test_run_returns_artifacts_dataclass(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_engineering_module, "PROCESSED_DIR", tmp_path)

    artifacts = run(_sample_validation(), CONFIG)

    assert isinstance(artifacts, FeatureEngineeringArtifacts)
    assert artifacts.x_train_path.exists()
    assert artifacts.x_test_path.exists()
    assert artifacts.y_train_path.exists()
    assert artifacts.y_test_path.exists()


def test_run_respects_test_size(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_engineering_module, "PROCESSED_DIR", tmp_path)

    validation = _sample_validation(n=20)

    artifacts = run(validation, CONFIG)

    X_train = pd.read_csv(artifacts.x_train_path)
    X_test = pd.read_csv(artifacts.x_test_path)

    assert len(X_train) + len(X_test) == len(validation.dataframe)
    assert len(X_test) == 4  # 20 rows * 0.2 test_size


def test_run_produces_matching_row_counts_between_features_and_target(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(feature_engineering_module, "PROCESSED_DIR", tmp_path)

    artifacts = run(_sample_validation(), CONFIG)

    X_train = pd.read_csv(artifacts.x_train_path)
    y_train = pd.read_csv(artifacts.y_train_path)
    X_test = pd.read_csv(artifacts.x_test_path)
    y_test = pd.read_csv(artifacts.y_test_path)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_run_does_not_leak_target_column_into_features(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_engineering_module, "PROCESSED_DIR", tmp_path)

    artifacts = run(_sample_validation(), CONFIG)

    X_train = pd.read_csv(artifacts.x_train_path)
    X_test = pd.read_csv(artifacts.x_test_path)

    assert "species" not in X_train.columns
    assert "species" not in X_test.columns


def test_run_is_reproducible_with_same_random_state(tmp_path, monkeypatch):
    df = _sample_df()

    first_dir = tmp_path / "first"
    monkeypatch.setattr(feature_engineering_module, "PROCESSED_DIR", first_dir)
    first = run(ValidationArtifacts(dataframe=df, report_path=Path("unused.json")), CONFIG)
    first_train = pd.read_csv(first.x_train_path)

    second_dir = tmp_path / "second"
    monkeypatch.setattr(feature_engineering_module, "PROCESSED_DIR", second_dir)
    second = run(ValidationArtifacts(dataframe=df, report_path=Path("unused.json")), CONFIG)
    second_train = pd.read_csv(second.x_train_path)

    pd.testing.assert_frame_equal(first_train, second_train)


def test_run_uses_test_size_and_random_state_from_config(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_engineering_module, "PROCESSED_DIR", tmp_path)

    config = {
        "schema": {"target_column": "species"},
        "training": {"test_size": 0.4, "random_state": 7},
    }

    artifacts = run(_sample_validation(n=50), config)

    X_test = pd.read_csv(artifacts.x_test_path)

    assert len(X_test) == 20  # 50 rows * 0.4 test_size
