"""
Tests for column name normalization utilities.
"""

import pandas as pd

from src.data.column_utils import normalize_column_name, normalize_columns


def test_normalize_column_name_handles_units_and_spaces():
    assert normalize_column_name("sepal length (cm)") == "sepal_length_cm"


def test_normalize_column_name_handles_mixed_case():
    assert normalize_column_name("SepalLength") == "sepallength"


def test_normalize_column_name_collapses_repeated_separators():
    assert normalize_column_name("Body Mass -- Index") == "body_mass_index"


def test_normalize_columns_renames_all_columns():
    df = pd.DataFrame(
        {
            "sepal length (cm)": [5.1],
            "species": ["setosa"],
        }
    )

    result = normalize_columns(df)

    assert list(result.columns) == ["sepal_length_cm", "species"]
