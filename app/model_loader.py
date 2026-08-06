"""
Model runtime loading for the FastAPI serving layer.

Owns everything about obtaining a ModelBundle: loading it fresh from
disk once at startup (load_model_bundle), and retrieving the
already-loaded instance during request handling via FastAPI's
dependency injection (get_model_bundle).
"""

import json
from dataclasses import dataclass
from typing import Any

import joblib
from fastapi import Request

from src.config.settings import ARTIFACT_DIR, MODEL_DIR

MODEL_FILENAME = "model.pkl"
METRICS_FILENAME = "metrics.json"


@dataclass(frozen=True)
class ModelBundle:
    """
    Everything the serving layer needs at request time: the fitted
    model, its training contract, and its evaluation metrics.
    """

    model: Any
    contract: dict
    metrics: dict


def load_model_bundle(contract: dict) -> ModelBundle:
    """
    Load the fitted model and evaluation metrics from disk.

    Args:
        contract: Already-loaded training contract — accepted as a
            parameter rather than re-read here, so training_report.json
            is read exactly once per app startup, not once per caller.

    Returns:
        A fully populated ModelBundle.
    """

    model = joblib.load(MODEL_DIR / MODEL_FILENAME)

    with open(ARTIFACT_DIR / METRICS_FILENAME) as file:
        metrics = json.load(file)

    return ModelBundle(model=model, contract=contract, metrics=metrics)


def get_model_bundle(request: Request) -> ModelBundle:
    """
    FastAPI dependency provider: retrieve the ModelBundle loaded at
    startup (lifespan) from application state.

    This is the single place any code reaches into `app.state` — route
    handlers depend on this function via Depends(get_model_bundle)
    instead of touching `request.app.state` directly, which also makes
    it trivial to override in tests via `app.dependency_overrides`.
    """

    return request.app.state.model_bundle
