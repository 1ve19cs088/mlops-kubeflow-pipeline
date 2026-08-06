"""
Feature engineering stage.

Splits a validated dataset into train/test feature and target sets,
persists each as a CSV artifact under data/processed/, and returns a
typed reference to those artifacts.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import PROCESSED_DIR
from src.data.validate_data import ValidationArtifacts
from src.utils.logger import get_logger

logger = get_logger(__name__)

X_TRAIN_FILENAME = "X_train.csv"
X_TEST_FILENAME = "X_test.csv"
Y_TRAIN_FILENAME = "y_train.csv"
Y_TEST_FILENAME = "y_test.csv"


@dataclass(frozen=True)
class FeatureEngineeringArtifacts:
    """
    Paths to the artifacts produced by the feature engineering stage.

    Mirrors the shape of a Kubeflow Pipeline component's Output[Dataset]
    parameters: downstream stages depend on this typed reference rather
    than on hardcoded filenames.
    """

    x_train_path: Path
    x_test_path: Path
    y_train_path: Path
    y_test_path: Path


def split_features_and_target(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature columns from the target column.

    Args:
        df: Validated dataset.
        target_column: Name of the target column.

    Returns:
        Tuple of (features, target).
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def run(validation: ValidationArtifacts, config: dict) -> FeatureEngineeringArtifacts:
    """
    Execute the feature engineering stage.

    Args:
        validation: Artifacts produced by the data validation stage.
        config: Parsed YAML configuration.

    Returns:
        FeatureEngineeringArtifacts referencing the persisted train/test splits.
    """

    logger.info("Starting feature engineering stage...")

    df = validation.dataframe
    target_column = config["schema"]["target_column"]
    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]

    X, y = split_features_and_target(df, target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    x_train_path = PROCESSED_DIR / X_TRAIN_FILENAME
    x_test_path = PROCESSED_DIR / X_TEST_FILENAME
    y_train_path = PROCESSED_DIR / Y_TRAIN_FILENAME
    y_test_path = PROCESSED_DIR / Y_TEST_FILENAME

    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    logger.info(
        f"Feature engineering complete: train={X_train.shape}, test={X_test.shape}"
    )

    return FeatureEngineeringArtifacts(
        x_train_path=x_train_path,
        x_test_path=x_test_path,
        y_train_path=y_train_path,
        y_test_path=y_test_path,
    )
