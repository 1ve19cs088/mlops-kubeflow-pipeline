"""
Dashboard page routes.

Every route calls the existing FastAPI service through ApiClient and
renders a template — no prediction/validation/training/evaluation
logic lives here, only presentation.
"""

import time
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates

from dashboard.api_client import (
    ApiClient,
    ApiUnavailableError,
    ApiValidationError,
    get_api_client,
)
from dashboard.dtype_utils import coerce_value, html_input_type

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter()


@router.get("/")
def home(request: Request, api_client: ApiClient = Depends(get_api_client)):
    health = None
    metadata = None
    metrics = None
    error = None

    try:
        health = api_client.get_health()
        metadata = api_client.get_metadata()
        metrics = api_client.get_metrics()
    except ApiUnavailableError as exc:
        error = str(exc)

    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "health": health,
            "metadata": metadata,
            "metrics": metrics,
            "error": error,
        },
    )


@router.get("/metrics")
def metrics_page(request: Request, api_client: ApiClient = Depends(get_api_client)):
    metrics = None
    error = None

    try:
        metrics = api_client.get_metrics()
    except ApiUnavailableError as exc:
        error = str(exc)

    return templates.TemplateResponse(
        request,
        "metrics.html",
        {"metrics": metrics, "error": error},
    )


def _predict_context(metadata, error, **overrides):
    context = {
        "metadata": metadata,
        "error": error,
        "input_types": {},
        "form_values": {},
        "prediction": None,
        "confidence": None,
        "response_time_ms": None,
        "predict_error": None,
    }
    if metadata:
        context["input_types"] = {
            name: html_input_type(metadata["feature_dtypes"][name])
            for name in metadata["feature_names"]
        }
    context.update(overrides)
    return context


@router.get("/predict")
def predict_form(request: Request, api_client: ApiClient = Depends(get_api_client)):
    metadata = None
    error = None

    try:
        metadata = api_client.get_metadata()
    except ApiUnavailableError as exc:
        error = str(exc)

    return templates.TemplateResponse(
        request, "predict.html", _predict_context(metadata, error)
    )


@router.post("/predict")
async def predict_submit(
    request: Request, api_client: ApiClient = Depends(get_api_client)
):
    metadata = None
    error = None

    try:
        metadata = api_client.get_metadata()
    except ApiUnavailableError as exc:
        error = str(exc)

    if not metadata:
        return templates.TemplateResponse(
            request, "predict.html", _predict_context(metadata, error)
        )

    form = await request.form()

    payload = {}
    form_values = {}
    for name in metadata["feature_names"]:
        dtype = metadata["feature_dtypes"][name]
        raw = form.get(name, "")
        form_values[name] = raw
        payload[name] = coerce_value(raw or "0", dtype)

    prediction = None
    confidence = None
    response_time_ms = None
    predict_error = None

    start = time.perf_counter()
    try:
        result = api_client.predict(payload)
        response_time_ms = (time.perf_counter() - start) * 1000
        prediction = result.get("prediction")
        confidence = result.get("confidence")
    except ApiValidationError as exc:
        predict_error = f"Invalid input: {exc.detail}"
    except ApiUnavailableError as exc:
        predict_error = f"API unavailable: {exc}"

    return templates.TemplateResponse(
        request,
        "predict.html",
        _predict_context(
            metadata,
            None,
            form_values=form_values,
            prediction=prediction,
            confidence=confidence,
            response_time_ms=response_time_ms,
            predict_error=predict_error,
        ),
    )
