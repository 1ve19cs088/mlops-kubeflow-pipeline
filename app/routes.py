"""
API routes for the model serving layer, mounted under /v1.
"""

import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.model_loader import ModelBundle, get_model_bundle
from app.schemas import (
    BatchPredictionResponse,
    HealthResponse,
    MetadataResponse,
    MetricsResponse,
    PredictionResponse,
)


def _to_dataframe(records: list[dict], feature_names: list[str]) -> pd.DataFrame:
    """
    Convert validated request records into a model-ready dataframe.

    Explicitly reorders columns to match feature_names — the order the
    model was trained on. A client's JSON field order should never be
    trusted to already match the training contract.
    """

    return pd.DataFrame(records)[feature_names]


def build_router(
    prediction_model: type[BaseModel], batch_model: type[BaseModel]
) -> APIRouter:
    """
    Build the /v1 API router.

    Constructed by a factory function rather than declared at module
    level: the /predict handlers are typed against prediction_model/
    batch_model, Pydantic models built dynamically from the training
    contract inside create_app() — not known at module import time.

    Args:
        prediction_model: Dynamically built single-record request model.
        batch_model: Dynamically built batch request model.

    Returns:
        A configured APIRouter.
    """

    router = APIRouter(prefix="/v1")

    @router.post("/predict", response_model=PredictionResponse)
    def predict(
        payload: prediction_model,
        model_bundle: ModelBundle = Depends(get_model_bundle),
    ) -> PredictionResponse:
        df = _to_dataframe(
            [payload.model_dump()], model_bundle.contract["feature_names"]
        )
        prediction = model_bundle.model.predict(df)[0]

        return PredictionResponse(prediction=prediction)

    @router.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(
        payload: batch_model,
        model_bundle: ModelBundle = Depends(get_model_bundle),
    ) -> BatchPredictionResponse:
        records = [record.model_dump() for record in payload.records]
        df = _to_dataframe(records, model_bundle.contract["feature_names"])
        predictions = model_bundle.model.predict(df)

        return BatchPredictionResponse(predictions=list(predictions))

    @router.get("/health", response_model=HealthResponse)
    def health(
        model_bundle: ModelBundle = Depends(get_model_bundle),
    ) -> HealthResponse:
        return HealthResponse(status="ok")

    @router.get("/metadata", response_model=MetadataResponse)
    def metadata(
        model_bundle: ModelBundle = Depends(get_model_bundle),
    ) -> MetadataResponse:
        contract = model_bundle.contract

        return MetadataResponse(
            algorithm=contract["algorithm"],
            framework=contract["framework"],
            model_version=contract["model_version"],
            trained_at=contract["trained_at"],
            feature_names=contract["feature_names"],
            feature_dtypes=contract["feature_dtypes"],
            target_column=contract["target_column"],
            class_labels=contract["class_labels"],
        )

    @router.get("/metrics", response_model=MetricsResponse)
    def metrics(
        model_bundle: ModelBundle = Depends(get_model_bundle),
    ) -> MetricsResponse:
        return MetricsResponse(**model_bundle.metrics)

    return router
