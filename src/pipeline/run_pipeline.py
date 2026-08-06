"""
Pipeline entry point.

This module orchestrates the MLOps pipeline.
"""

from src.config.settings import CONFIG_DIR
from src.config.yaml_loader import load_yaml
from src.data.ingest_data import run as ingest_data
from src.data.validate_data import run as validate_data
from src.features.feature_engineering import run as engineer_features
from src.models.evaluate import run as evaluate_model
from src.models.train import run as train_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """
    Main pipeline execution.
    """

    config_path = CONFIG_DIR / "iris.yaml"

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

    training = train_model(features, config)

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

    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()