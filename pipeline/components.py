"""
Kubeflow Pipelines v2 component definitions.

Each component is a thin wrapper: construct the same dataclass the
local orchestrator (src/pipeline/run_pipeline.py) already uses,
pointing at KFP-managed artifact paths instead of fixed local
directories, call the unchanged src/ business logic, then copy the
result into KFP's output artifact locations. The actual computation
lives entirely in src/ — nothing is reimplemented here.
"""

from kfp import dsl
from kfp.dsl import Artifact, ClassificationMetrics, Dataset, Input, Metrics, Model, Output

PIPELINE_IMAGE = "mlops-kubeflow-pipeline-pipeline:v1"


@dsl.component(base_image=PIPELINE_IMAGE)
def ingest_data_component(config: dict, output_dataset: Output[Dataset]):
    from src.data.ingest_data import run as ingest_data

    df = ingest_data(config)
    df.to_csv(output_dataset.path, index=False)


@dsl.component(base_image=PIPELINE_IMAGE)
def validate_data_component(
    input_dataset: Input[Dataset],
    config: dict,
    output_dataset: Output[Dataset],
    validation_metrics: Output[Metrics],
):
    import json

    import pandas as pd

    from src.data.validate_data import run as validate_data

    df = pd.read_csv(input_dataset.path)
    validation = validate_data(df, config)  # raises + fails the task if invalid

    validation.dataframe.to_csv(output_dataset.path, index=False)

    report = json.loads(validation.report_path.read_text())
    validation_metrics.log_metric("row_count", report["row_count"])
    validation_metrics.log_metric("column_count", report["column_count"])
    validation_metrics.log_metric("valid", 1.0 if report["valid"] else 0.0)


@dsl.component(base_image=PIPELINE_IMAGE)
def feature_engineering_component(
    input_dataset: Input[Dataset],
    config: dict,
    x_train: Output[Dataset],
    x_test: Output[Dataset],
    y_train: Output[Dataset],
    y_test: Output[Dataset],
):
    import shutil
    from pathlib import Path

    import pandas as pd

    from src.data.validate_data import ValidationArtifacts
    from src.features.feature_engineering import run as engineer_features

    # KFP marshals `dict` pipeline parameters through protobuf Struct,
    # which has no distinct int type — every number arrives as a float.
    # random_state must be a real int for sklearn's strict validation;
    # this is a KFP parameter-marshaling quirk, not a src/ bug, so the
    # fix belongs in this glue layer, not in feature_engineering.py.
    config["training"]["random_state"] = int(config["training"]["random_state"])

    df = pd.read_csv(input_dataset.path)
    validation = ValidationArtifacts(
        dataframe=df, report_path=Path("/tmp/unused_validation_report.json")
    )

    features = engineer_features(validation, config)

    shutil.copy(features.x_train_path, x_train.path)
    shutil.copy(features.x_test_path, x_test.path)
    shutil.copy(features.y_train_path, y_train.path)
    shutil.copy(features.y_test_path, y_test.path)


@dsl.component(base_image=PIPELINE_IMAGE)
def train_component(
    x_train: Input[Dataset],
    x_test: Input[Dataset],
    y_train: Input[Dataset],
    y_test: Input[Dataset],
    config: dict,
    model: Output[Model],
    training_report: Output[Artifact],
):
    import json
    import shutil
    from pathlib import Path

    from src.features.feature_engineering import FeatureEngineeringArtifacts
    from src.models.train import run as train_run

    # Same KFP dict-parameter float-coercion quirk as in
    # feature_engineering_component — see comment there.
    config["training"]["random_state"] = int(config["training"]["random_state"])

    features = FeatureEngineeringArtifacts(
        x_train_path=Path(x_train.path),
        x_test_path=Path(x_test.path),
        y_train_path=Path(y_train.path),
        y_test_path=Path(y_test.path),
    )

    training = train_run(features, config)

    shutil.copy(training.model_path, model.path)
    shutil.copy(training.report_path, training_report.path)

    report = json.loads(training.report_path.read_text())
    model.metadata.update(
        {
            "algorithm": report["algorithm"],
            "framework": report["framework"],
            "model_version": report["model_version"],
            "trained_at": report["trained_at"],
            "feature_names": report["feature_names"],
            "feature_dtypes": report["feature_dtypes"],
            "target_column": report["target_column"],
            "class_labels": report["class_labels"],
        }
    )


@dsl.component(base_image=PIPELINE_IMAGE)
def evaluate_component(
    model: Input[Model],
    training_report: Input[Artifact],
    x_train: Input[Dataset],
    x_test: Input[Dataset],
    y_train: Input[Dataset],
    y_test: Input[Dataset],
    config: dict,
    classification_metrics: Output[ClassificationMetrics],
    scalar_metrics: Output[Metrics],
    predictions: Output[Dataset],
):
    import shutil
    from pathlib import Path

    from src.features.feature_engineering import FeatureEngineeringArtifacts
    from src.models.evaluate import run as evaluate_run
    from src.models.train import TrainingArtifacts

    training = TrainingArtifacts(
        model_path=Path(model.path), report_path=Path(training_report.path)
    )
    features = FeatureEngineeringArtifacts(
        x_train_path=Path(x_train.path),
        x_test_path=Path(x_test.path),
        y_train_path=Path(y_train.path),
        y_test_path=Path(y_test.path),
    )

    evaluation = evaluate_run(training, features, config)

    shutil.copy(evaluation.predictions_path, predictions.path)

    for split_name, split_metrics in (
        ("train", evaluation.metrics["train"]),
        ("test", evaluation.metrics["test"]),
    ):
        for key in ("accuracy", "precision", "recall", "f1_score"):
            scalar_metrics.log_metric(f"{split_name}_{key}", split_metrics[key])

    # KFP's ClassificationMetrics has a purpose-built confusion matrix
    # renderer — this is the "big mapping insight" from the FastAPI
    # Kubeflow-mapping discussion, now actually implemented: no need to
    # hand-generate a PNG for KFP's own UI, only for external reporting
    # (evaluate.run() still writes confusion_matrix.png/.csv locally
    # inside the container regardless, unchanged).
    confusion_matrix = evaluation.metrics["test"]["confusion_matrix"]
    classification_metrics.log_confusion_matrix(
        confusion_matrix["labels"], confusion_matrix["matrix"]
    )
