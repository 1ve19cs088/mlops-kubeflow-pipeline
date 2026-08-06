"""
Pipeline entry point.

This module orchestrates the MLOps pipeline.
"""

import os
import time

from src.config.settings import CONFIG_DIR
from src.config.yaml_loader import load_yaml
from src.data.ingest_data import run as ingest_data
from src.data.validate_data import run as validate_data
from src.features.feature_engineering import run as engineer_features
from src.models.evaluate import run as evaluate_model
from src.models.train import run as train_model
from src.tracking.mlflow_tracking import log_run as log_mlflow_run
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_FILENAME = "iris.yaml"


def main(config_filename: str = DEFAULT_CONFIG_FILENAME):
    """
    Main pipeline execution.

    Args:
        config_filename: Name of the YAML config file under configs/
            to run the pipeline with.
    """

    config_path = CONFIG_DIR / config_filename

    config = load_yaml(config_path)

    logger.info("Configuration loaded successfully.")

    logger.info(f"Dataset: {config['dataset']['name']}")

    logger.info("========== DATA INGESTION STAGE ==========")

    df = ingest_data(config)

    logger.info(f"Pipeline received dataframe with shape: {df.shape}")

    logger.info("========== DATA VALIDATION STAGE ==========")

    validation = validate_data(df, config)

    logger.info("========== FEATURE ENGINEERING STAGE ==========")

    features = engineer_features(validation, config)

    logger.info(f"Feature engineering artifacts: {features}")

    logger.info("========== MODEL TRAINING STAGE ==========")

    training_start = time.perf_counter()
    training = train_model(features, config)
    training_duration_seconds = time.perf_counter() - training_start

    logger.info(f"Training artifacts: {training}")

    logger.info("========== MODEL EVALUATION STAGE ==========")

    evaluation = evaluate_model(training, features, config)

    train_accuracy = evaluation.metrics["train"]["accuracy"]
    test_accuracy = evaluation.metrics["test"]["accuracy"]
    overfitting_gap = train_accuracy - test_accuracy

    logger.info(
        f"Evaluation summary — train accuracy: {train_accuracy:.4f}, "
        f"test accuracy: {test_accuracy:.4f}, "
        f"overfitting gap: {overfitting_gap:.4f}"
    )
    logger.info(f"Evaluation artifacts: {evaluation}")

    logger.info("========== MLFLOW TRACKING STAGE ==========")

    run_id = log_mlflow_run(
        training, features, evaluation, config, training_duration_seconds
    )

    logger.info(f"MLflow run ID: {run_id}")

    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main(os.environ.get("CONFIG_FILE", DEFAULT_CONFIG_FILENAME))