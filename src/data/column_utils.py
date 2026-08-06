"""
Column name normalization utilities.

Ensures dataset column names are consistent regardless of the upstream
data source's naming conventions (e.g. sklearn's "sepal length (cm)"
vs a CSV's "SepalLength"), so config and downstream stages can rely on
a single, predictable naming convention.
"""

import re

import pandas as pd


def normalize_column_name(name: str) -> str:
    """
    Convert an arbitrary column name into a clean snake_case identifier.

    Args:
        name: Raw column name.

    Returns:
        Normalized snake_case column name.
    """

    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")

    return name


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with all column names normalized to snake_case.

    Args:
        df: Input dataframe.

    Returns:
        Dataframe with normalized column names.
    """

    return df.rename(
        columns={column: normalize_column_name(column) for column in df.columns}
    )
