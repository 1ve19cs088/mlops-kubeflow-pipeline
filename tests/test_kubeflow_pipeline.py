"""
Tests for the Kubeflow Pipeline definition. Compilation-only checks
run fast (no Docker); the full local execution test is slow (builds
real containers) and shares the `docker` marker's exclusion from the
default test run — see pytest.ini.

Run explicitly with: pytest -m docker -v
"""

import pytest
import yaml

from pipeline.compile_pipeline import compile_pipeline

EXPECTED_COMPONENTS = {
    "comp-ingest-data-component",
    "comp-validate-data-component",
    "comp-feature-engineering-component",
    "comp-train-component",
    "comp-evaluate-component",
}


def test_pipeline_compiles_with_expected_components():
    output_path = compile_pipeline()

    assert output_path.exists()

    spec = yaml.safe_load(output_path.read_text())

    assert EXPECTED_COMPONENTS.issubset(spec["components"].keys())


@pytest.mark.docker
def test_pipeline_runs_locally_and_produces_expected_outputs(tmp_path):
    import kfp.local

    from pipeline.kubeflow_pipeline import mlops_pipeline
    from src.config.settings import CONFIG_DIR
    from src.config.yaml_loader import load_yaml

    kfp.local.init(runner=kfp.local.DockerRunner(), pipeline_root=str(tmp_path))

    config = load_yaml(CONFIG_DIR / "iris.yaml")

    # kfp.local raises RuntimeError on any component failure, so
    # simply reaching this line means the whole DAG succeeded. Confirm
    # the expected output files actually landed too.
    mlops_pipeline(config=config)

    run_dirs = list(tmp_path.glob("mlops-kubeflow-pipeline-*"))
    assert len(run_dirs) == 1

    run_dir = run_dirs[0]
    assert (run_dir / "train-component" / "model").exists()
    assert (run_dir / "evaluate-component" / "predictions").exists()
