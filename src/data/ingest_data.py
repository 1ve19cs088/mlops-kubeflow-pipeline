"""
Data ingestion stage.
"""

import pandas as pd

from src.data.dataset_loader import load_dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run(config: dict) -> pd.DataFrame:
    """
    Execute the ingestion stage.

    Args:
        config: Parsed YAML configuration.

    Returns:
        Loaded dataset.
    """

    logger.info("Starting data ingestion stage...")

    df = load_dataset(config)

    logger.info(f"Dataset loaded successfully: {df.shape}")

    return df