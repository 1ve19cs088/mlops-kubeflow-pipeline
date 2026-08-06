"""
Executes the compiled Kubeflow Pipeline locally via kfp.local's
DockerRunner — each component runs as a real Docker container using
our own pipeline image (no external registry pulls), while still
exercising the genuine KFP v2 component/artifact/DAG machinery.
"""

import os

import kfp.local

from src.config.settings import CONFIG_DIR
from src.config.yaml_loader import load_yaml

from pipeline.kubeflow_pipeline import mlops_pipeline

DEFAULT_CONFIG_FILENAME = "iris.yaml"


if __name__ == "__main__":
    kfp.local.init(runner=kfp.local.DockerRunner(), pipeline_root="./pipeline/local_outputs")

    config_filename = os.environ.get("CONFIG_FILE", DEFAULT_CONFIG_FILENAME)
    config = load_yaml(CONFIG_DIR / config_filename)

    mlops_pipeline(config=config)
