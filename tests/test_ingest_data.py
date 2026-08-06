"""
Tests for the data ingestion stage.
"""

import pandas as pd
import pytest

from src.data.ingest_data import run

CONFIG = {
    "dataset": {
        "name": "iris",
        "source": {"type": "sklearn", "loader": "iris"},
    }
}


def test_run_returns_dataframe():
    df = run(CONFIG)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 150


def test_run_returns_normalized_columns():
    df = run(CONFIG)

    assert "species" in df.columns
    assert "sepal_length_cm" in df.columns


def test_run_propagates_dataset_loader_errors():
    bad_config = {"dataset": {"source": {"type": "not_a_real_source"}}}

    with pytest.raises(ValueError):
        run(bad_config)
