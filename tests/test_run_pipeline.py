"""
Integration test for the local pipeline orchestrator (main()).

Monkeypatches each stage module's own imported output-directory
constants (the same isolation pattern already used in every other
stage's test file) so this runs the real, full ingest -> validate ->
feature engineering -> train -> evaluate -> MLflow tracking chain
without touching the project's actual data/processed/, model/,
artifacts/, or mlflow.db.
"""

import src.data.validate_data as validate_data_module
import src.features.feature_engineering as feature_engineering_module
import src.models.evaluate as evaluate_module
import src.models.train as train_module
import src.tracking.mlflow_tracking as mlflow_tracking_module
from src.pipeline.run_pipeline import main


def test_main_runs_full_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(validate_data_module, "ARTIFACT_DIR", tmp_path / "validation_artifacts")
    monkeypatch.setattr(feature_engineering_module, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path / "model")
    monkeypatch.setattr(train_module, "ARTIFACT_DIR", tmp_path / "train_artifacts")
    monkeypatch.setattr(evaluate_module, "ARTIFACT_DIR", tmp_path / "eval_artifacts")
    monkeypatch.setattr(
        mlflow_tracking_module,
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{tmp_path}/mlflow_test.db",
    )
    monkeypatch.setattr(
        mlflow_tracking_module,
        "MLFLOW_ARTIFACT_ROOT",
        f"file://{tmp_path}/mlruns",
    )

    main()

    assert (tmp_path / "model" / "model.pkl").exists()
    assert (tmp_path / "train_artifacts" / "training_report.json").exists()
    assert (tmp_path / "eval_artifacts" / "metrics.json").exists()
    assert (tmp_path / "eval_artifacts" / "predictions.csv").exists()
