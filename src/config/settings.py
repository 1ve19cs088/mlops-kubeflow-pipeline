"""
Project settings and directory paths.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Project Root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

CONFIG_DIR = PROJECT_ROOT / "configs"

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

DATASET_DIR = DATA_DIR / "datasets"

PROCESSED_DIR = DATA_DIR / "processed"

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

MODEL_DIR = PROJECT_ROOT / "model"

# ---------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

# ---------------------------------------------------------------------
# Create folders automatically
# ---------------------------------------------------------------------

for directory in [
    CONFIG_DIR,
    DATASET_DIR,
    PROCESSED_DIR,
    MODEL_DIR,
    ARTIFACT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)