"""
Dashboard page routes.

Every route calls the existing FastAPI service through ApiClient and
renders a template — no prediction/validation/training/evaluation
logic lives here, only presentation.
"""

import io
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates

from dashboard.api_client import (
    ApiClient,
    ApiUnavailableError,
    ApiValidationError,
    get_api_client,
)
from dashboard.deployment_info import get_deployment_info
from dashboard.dtype_utils import coerce_value, html_input_type
from dashboard.mlflow_client import MlflowRegistryClient, get_mlflow_client
from dashboard.status_info import get_environment_status, get_github_actions_badge_url

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


def _batch_context(metadata, error, **overrides):
    context = {
        "metadata": metadata,
        "error": error,
        "preview_rows": None,
        "preview_columns": None,
        "batch_error": None,
    }
    context.update(overrides)
    return context


@router.get("/batch")
def batch_form(request: Request, api_client: ApiClient = Depends(get_api_client)):
    metadata = None
    error = None

    try:
        metadata = api_client.get_metadata()
    except ApiUnavailableError as exc:
        error = str(exc)

    return templates.TemplateResponse(
        request, "batch.html", _batch_context(metadata, error)
    )


@router.post("/batch")
async def batch_submit(
    request: Request,
    file: UploadFile = File(...),
    action: str = Form("preview"),
    api_client: ApiClient = Depends(get_api_client),
):
    metadata = None
    error = None

    try:
        metadata = api_client.get_metadata()
    except ApiUnavailableError as exc:
        error = str(exc)

    if not metadata:
        return templates.TemplateResponse(
            request, "batch.html", _batch_context(metadata, error)
        )

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    batch_error = None
    result_df = None

    missing = [name for name in metadata["feature_names"] if name not in df.columns]
    if missing:
        batch_error = f"CSV is missing required columns: {', '.join(missing)}"
    else:
        records = [
            {
                name: coerce_value(row[name], metadata["feature_dtypes"][name])
                for name in metadata["feature_names"]
            }
            for _, row in df.iterrows()
        ]

        try:
            result = api_client.predict_batch(records)
            result_df = df.copy()
            result_df["prediction"] = result["predictions"]
        except ApiValidationError as exc:
            batch_error = f"Invalid input: {exc.detail}"
        except ApiUnavailableError as exc:
            batch_error = f"API unavailable: {exc}"

    if result_df is not None and action == "download":
        buffer = io.StringIO()
        result_df.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    preview_rows = None
    preview_columns = None
    if result_df is not None:
        preview_columns = list(result_df.columns)
        preview_rows = result_df.head(20).values.tolist()

    return templates.TemplateResponse(
        request,
        "batch.html",
        _batch_context(
            metadata,
            None,
            preview_rows=preview_rows,
            preview_columns=preview_columns,
            batch_error=batch_error,
        ),
    )


@router.get("/deployment")
def deployment_page(request: Request):
    info = get_deployment_info()

    return templates.TemplateResponse(request, "deployment.html", {"info": info})


@router.get("/status")
def status_page(request: Request, api_client: ApiClient = Depends(get_api_client)):
    api_ok = False
    model_ok = False
    model_info = None
    api_error = None

    try:
        api_client.get_health()
        api_ok = True
    except ApiUnavailableError as exc:
        api_error = str(exc)

    try:
        model_info = api_client.get_metadata()
        model_ok = True
    except ApiUnavailableError:
        pass

    environment = get_environment_status()

    return templates.TemplateResponse(
        request,
        "status.html",
        {
            "api_ok": api_ok,
            "model_ok": model_ok,
            "model_info": model_info,
            "api_error": api_error,
            "docker": environment["docker"],
            "kubernetes": environment["kubernetes"],
            "github_badge_url": get_github_actions_badge_url(),
        },
    )


def _format_timestamp(epoch_millis: int) -> str:
    return datetime.fromtimestamp(epoch_millis / 1000, tz=timezone.utc).isoformat()


def _build_model_row(mlflow_client: MlflowRegistryClient, model) -> dict:
    versions = mlflow_client.get_model_versions(model.name)
    latest = versions[0] if versions else None

    latest_accuracy = None
    if latest is not None:
        metrics = mlflow_client.get_run_metrics(latest.run_id)
        latest_accuracy = metrics.get("test_accuracy")

    return {
        "name": model.name,
        "latest_version": latest.version if latest is not None else None,
        "current_stage": latest.current_stage if latest is not None else None,
        "latest_accuracy": latest_accuracy,
        "created_time": _format_timestamp(model.creation_timestamp),
        "num_versions": len(versions),
    }


@router.get("/models")
def models_list(
    request: Request,
    mlflow_client: MlflowRegistryClient = Depends(get_mlflow_client),
):
    models = mlflow_client.get_registered_models()
    rows = [_build_model_row(mlflow_client, model) for model in models]

    return templates.TemplateResponse(request, "models.html", {"models": rows})


@router.get("/models/{model_name}")
def model_detail(request: Request, model_name: str):
    return templates.TemplateResponse(
        request, "model_detail.html", {"model_name": model_name}
    )
