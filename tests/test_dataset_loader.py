"""
Tests for dataset loading.
"""

import pandas as pd
import pytest

from src.data.dataset_loader import load_dataset

IRIS_CONFIG = {
    "dataset": {
        "name": "iris",
        "source": {"type": "sklearn", "loader": "iris"},
    }
}


def test_load_dataset_returns_expected_iris_dataframe():
    df = load_dataset(IRIS_CONFIG)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (150, 5)
    assert "species" in df.columns
    assert "target" not in df.columns
    assert set(df["species"].unique()) == {"setosa", "versicolor", "virginica"}


def test_load_dataset_normalizes_column_names():
    df = load_dataset(IRIS_CONFIG)

    assert "sepal_length_cm" in df.columns
    assert "sepal length (cm)" not in df.columns


def test_load_dataset_rejects_unsupported_sklearn_loader():
    config = {"dataset": {"source": {"type": "sklearn", "loader": "not_a_real_dataset"}}}

    with pytest.raises(ValueError, match="Unsupported sklearn dataset"):
        load_dataset(config)


def test_load_dataset_rejects_unsupported_source_type():
    config = {"dataset": {"source": {"type": "not_a_real_source"}}}

    with pytest.raises(ValueError, match="Unsupported dataset source"):
        load_dataset(config)
