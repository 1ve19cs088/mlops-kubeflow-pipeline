"""
Training contract loading for the FastAPI serving layer.

Reading training_report.json is deliberately NOT a side effect of
importing this module — it only happens when load_training_contract()
is explicitly called (from create_app()). Importing app.contract on
its own never touches the filesystem.
"""

import json

from src.config.settings import ARTIFACT_DIR

TRAINING_REPORT_FILENAME = "training_report.json"


def load_training_contract() -> dict:
    """
    Load the model's input contract written by the training stage.

    Returns:
        Parsed training_report.json contents.
    """

    with open(ARTIFACT_DIR / TRAINING_REPORT_FILENAME) as file:
        return json.load(file)
