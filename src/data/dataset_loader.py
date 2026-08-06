"""
Dataset loading module.

Responsible for loading datasets from different sources.
"""

import pandas as pd
from sklearn.datasets import load_iris

from src.data.column_utils import normalize_columns


def load_dataset(config: dict) -> pd.DataFrame:
    """
    Load dataset based on configuration.

    Args:
        config: Parsed YAML configuration.

    Returns:
        Pandas DataFrame.
    """

    source = config["dataset"]["source"]["type"]

    if source == "sklearn":

        loader = config["dataset"]["source"]["loader"]

        if loader == "iris":

            iris = load_iris(as_frame=True)

            df = iris.frame.copy()

            df["species"] = df["target"].map(
                {
                    0: "setosa",
                    1: "versicolor",
                    2: "virginica",
                }
            )

            df.drop(columns=["target"], inplace=True)

            return normalize_columns(df)

        raise ValueError(f"Unsupported sklearn dataset: {loader}")

    raise ValueError(f"Unsupported dataset source: {source}")