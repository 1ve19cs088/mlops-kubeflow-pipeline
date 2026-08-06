"""
Data validation stage.

Verifies that an ingested dataset matches the schema declared in
configuration before it is allowed to flow into feature engineering.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config.settings import ARTIFACT_DIR
from src.utils.logger import get_logger
from src.utils.report_writer import save_json_report

logger = get_logger(__name__)

VALIDATION_REPORT_FILENAME = "validation_report.json"


class DataValidationError(Exception):
    """Raised when a dataset fails schema validation."""


@dataclass(frozen=True)
class ValidationArtifacts:
    """
    Artifacts produced by the data validation stage.

    `dataframe` is carried in-memory (validation doesn't transform the
    data, so there's nothing new to persist there); `report_path`
    references the validation report written to disk. In a Kubeflow
    Pipeline component, `dataframe` would itself become an
    Output[Dataset] path rather than a live object, since components
    can't share Python memory across containers — this local stand-in
    keeps it in-memory for simplicity until that migration happens.
    """

    dataframe: pd.DataFrame
    report_path: Path


def _check_missing_columns(df: pd.DataFrame, expected_columns: list[str]) -> list[str]:
    return [column for column in expected_columns if column not in df.columns]


def _check_missing_values(df: pd.DataFrame) -> dict[str, int]:
    counts = df.isnull().sum()
    return {column: int(count) for column, count in counts.items() if count > 0}


def _check_unexpected_labels(
    df: pd.DataFrame, target_column: str, valid_labels: list[str]
) -> list[str]:
    if target_column not in df.columns:
        return []

    actual_labels = set(df[target_column].unique())
    return sorted(actual_labels - set(valid_labels))


def validate_schema(df: pd.DataFrame, config: dict) -> dict:
    """
    Validate a dataframe against the schema declared in configuration.

    Args:
        df: Ingested dataset.
        config: Parsed YAML configuration.

    Returns:
        Validation report dictionary.
    """

    schema = config["schema"]

    expected_columns = schema["expected_columns"]
    target_column = schema["target_column"]
    valid_labels = schema.get("valid_labels", [])

    missing_columns = _check_missing_columns(df, expected_columns)
    missing_values = _check_missing_values(df)
    unexpected_labels = _check_unexpected_labels(df, target_column, valid_labels)

    is_valid = not (missing_columns or missing_values or unexpected_labels)

    return {
        "valid": is_valid,
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_columns": missing_columns,
        "missing_values": missing_values,
        "unexpected_labels": unexpected_labels,
    }


def run(df: pd.DataFrame, config: dict) -> ValidationArtifacts:
    """
    Execute the data validation stage.

    Args:
        df: Ingested dataset.
        config: Parsed YAML configuration.

    Returns:
        ValidationArtifacts referencing the validated dataframe and report.

    Raises:
        DataValidationError: If the dataset fails schema validation.
    """

    logger.info("Starting data validation stage...")

    report = validate_schema(df, config)

    report_path = save_json_report(ARTIFACT_DIR, VALIDATION_REPORT_FILENAME, report)

    if not report["valid"]:
        logger.error(f"Data validation failed: {report}")
        raise DataValidationError(f"Dataset failed schema validation: {report}")

    logger.info(f"Data validation passed: {report}")

    return ValidationArtifacts(dataframe=df, report_path=report_path)
