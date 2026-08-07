"""
Dashboard page routes.

Every route calls the existing FastAPI service through ApiClient and
renders a template — no prediction/validation/training/evaluation
logic lives here, only presentation.
"""

import base64
import io
import json
import mimetypes
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
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


def _build_version_row(mlflow_client: MlflowRegistryClient, version) -> dict:
    metrics = mlflow_client.get_run_metrics(version.run_id)
    params = mlflow_client.get_run_parameters(version.run_id)

    return {
        "version": version.version,
        "current_stage": version.current_stage,
        "accuracy": metrics.get("test_accuracy"),
        "precision": metrics.get("test_precision"),
        "recall": metrics.get("test_recall"),
        "f1_score": metrics.get("test_f1_score"),
        "training_duration_seconds": metrics.get("training_duration_seconds"),
        "algorithm": params.get("algorithm"),
        "dataset": params.get("dataset"),
        "created_time": _format_timestamp(version.creation_timestamp),
        "run_id": version.run_id,
    }


@router.get("/models/{model_name}")
def model_detail(
    request: Request,
    model_name: str,
    mlflow_client: MlflowRegistryClient = Depends(get_mlflow_client),
):
    versions = mlflow_client.get_model_versions(model_name)
    rows = [_build_version_row(mlflow_client, version) for version in versions]

    return templates.TemplateResponse(
        request,
        "model_detail.html",
        {"model_name": model_name, "versions": rows},
    )


PREVIEWABLE_JSON_SUFFIXES = (".json",)
PREVIEWABLE_TEXT_SUFFIXES = (".txt",)
PREVIEWABLE_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif")


def _build_artifact_previews(
    mlflow_client: MlflowRegistryClient, run_id: str, model_name: str, version: str
) -> list[dict]:
    previews = []

    for artifact in mlflow_client.list_run_artifacts(run_id):
        if artifact.is_dir:
            continue

        lower_path = artifact.path.lower()
        entry = {
            "path": artifact.path,
            "kind": "download",
            "download_url": (
                f"/models/{model_name}/versions/{version}/artifacts/{artifact.path}"
            ),
        }

        try:
            if lower_path.endswith(PREVIEWABLE_JSON_SUFFIXES):
                data = mlflow_client.get_artifact_bytes(run_id, artifact.path)
                entry["kind"] = "json"
                entry["content"] = json.loads(data.decode("utf-8"))
            elif lower_path.endswith(PREVIEWABLE_TEXT_SUFFIXES):
                data = mlflow_client.get_artifact_bytes(run_id, artifact.path)
                entry["kind"] = "text"
                entry["content"] = data.decode("utf-8")
            elif lower_path.endswith(PREVIEWABLE_IMAGE_SUFFIXES):
                data = mlflow_client.get_artifact_bytes(run_id, artifact.path)
                mime = mimetypes.guess_type(artifact.path)[0] or "application/octet-stream"
                entry["kind"] = "image"
                entry["data_uri"] = (
                    f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
                )
        except Exception:
            entry["kind"] = "download"

        previews.append(entry)

    return previews


@router.get("/models/{model_name}/versions/{version}")
def model_version_detail(
    request: Request,
    model_name: str,
    version: str,
    mlflow_client: MlflowRegistryClient = Depends(get_mlflow_client),
):
    versions = mlflow_client.get_model_versions(model_name)
    matched = next((v for v in versions if str(v.version) == version), None)

    if matched is None:
        return templates.TemplateResponse(
            request,
            "version_detail.html",
            {"model_name": model_name, "version": version, "found": False},
            status_code=404,
        )

    return templates.TemplateResponse(
        request,
        "version_detail.html",
        {
            "model_name": model_name,
            "version": matched.version,
            "current_stage": matched.current_stage,
            "run_id": matched.run_id,
            "created_time": _format_timestamp(matched.creation_timestamp),
            "metrics": mlflow_client.get_run_metrics(matched.run_id),
            "params": mlflow_client.get_run_parameters(matched.run_id),
            "artifacts": _build_artifact_previews(
                mlflow_client, matched.run_id, model_name, matched.version
            ),
            "found": True,
        },
    )


@router.get("/models/{model_name}/versions/{version}/artifacts/{artifact_path:path}")
def download_artifact(
    model_name: str,
    version: str,
    artifact_path: str,
    mlflow_client: MlflowRegistryClient = Depends(get_mlflow_client),
):
    versions = mlflow_client.get_model_versions(model_name)
    matched = next((v for v in versions if str(v.version) == version), None)
    if matched is None:
        raise HTTPException(status_code=404, detail="Model version not found")

    try:
        content = mlflow_client.get_artifact_bytes(matched.run_id, artifact_path)
    except Exception:
        raise HTTPException(status_code=404, detail="Artifact not found")

    media_type = mimetypes.guess_type(artifact_path)[0] or "application/octet-stream"
    filename = artifact_path.rsplit("/", 1)[-1]

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
