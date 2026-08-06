"""
Compiles the Kubeflow Pipeline into its IR YAML — the portable
artifact that could be uploaded to any real KFP install or Vertex AI
Pipelines, independent of how it's executed here locally.
"""

from pathlib import Path

from kfp import compiler

from pipeline.kubeflow_pipeline import mlops_pipeline

OUTPUT_PATH = Path(__file__).resolve().parent / "compiled" / "mlops_pipeline.yaml"


def compile_pipeline() -> Path:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(mlops_pipeline, str(OUTPUT_PATH))
    return OUTPUT_PATH


if __name__ == "__main__":
    path = compile_pipeline()
    print(f"Compiled pipeline written to: {path}")
