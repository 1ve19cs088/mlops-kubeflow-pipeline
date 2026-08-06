"""
YAML configuration loader.
"""

from pathlib import Path
import yaml


def load_yaml(path: Path) -> dict:
    """
    Load a YAML configuration file.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed configuration dictionary.
    """

    with open(path, "r") as file:
        return yaml.safe_load(file)