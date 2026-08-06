"""
The Kubeflow Pipeline definition: wires the components in
pipeline/components.py into the same DAG src/pipeline/run_pipeline.py
already runs locally — ingest -> validate -> feature engineering ->
train -> evaluate.
"""

from kfp import dsl

from pipeline.components import (
    evaluate_component,
    feature_engineering_component,
    ingest_data_component,
    train_component,
    validate_data_component,
)


@dsl.pipeline(
    name="mlops-kubeflow-pipeline",
    description="Ingest -> Validate -> Feature Engineering -> Train -> Evaluate",
)
def mlops_pipeline(config: dict):
    ingest_task = ingest_data_component(config=config)

    validate_task = validate_data_component(
        input_dataset=ingest_task.outputs["output_dataset"],
        config=config,
    )

    features_task = feature_engineering_component(
        input_dataset=validate_task.outputs["output_dataset"],
        config=config,
    )

    train_task = train_component(
        x_train=features_task.outputs["x_train"],
        x_test=features_task.outputs["x_test"],
        y_train=features_task.outputs["y_train"],
        y_test=features_task.outputs["y_test"],
        config=config,
    )

    evaluate_component(
        model=train_task.outputs["model"],
        training_report=train_task.outputs["training_report"],
        x_train=features_task.outputs["x_train"],
        x_test=features_task.outputs["x_test"],
        y_train=features_task.outputs["y_train"],
        y_test=features_task.outputs["y_test"],
        config=config,
    )
