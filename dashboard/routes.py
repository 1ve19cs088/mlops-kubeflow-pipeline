"""
Dashboard page routes.

Prediction/batch-prediction routes call the existing FastAPI serving
API through ApiClient. Model-registry information (Home, Metrics, and
the Model Registry pages) is read through MlflowRegistryClient
instead — MLflow, not the API's local artifact files, is this
dashboard's source of truth for anything about a trained model. No
prediction/validation/training/evaluation logic lives here, only
presentation.
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
from dashboard.deployment_pipeline_status import get_deployment_pipeline_status
from dashboard.dtype_utils import coerce_value, html_input_type
from dashboard.mlflow_client import MlflowRegistryClient, get_mlflow_client
from dashboard.status_info import get_environment_status, get_github_actions_badge_url
from deployment.service import DeploymentService, get_deployment_service

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter()


def _latest_model_and_version(mlflow_client: MlflowRegistryClient, models: list):
    """
    The most recently updated registered model (models is expected to
    already be sorted newest-first by get_registered_models()) and its
    highest version number, if any model is registered at all.
    """

    if not models:
        return None, None

    latest_model_name = models[0].name
    return latest_model_name, mlflow_client.get_latest_version(latest_model_name)


def _build_registry_summary(mlflow_client: MlflowRegistryClient) -> dict:
    """
    Registry-wide counts plus a summary of the most recently updated
    model — shared by the Home and System Status pages so "which
    model is latest" is resolved in exactly one place.
    """

    models = mlflow_client.get_registered_models()
    latest_model_name, latest_version = _latest_model_and_version(mlflow_client, models)

    latest_stage = None
    latest_accuracy = None
    latest_created_time = None
    if latest_version is not None:
        latest_stage = latest_version.current_stage
        latest_created_time = _format_timestamp(latest_version.creation_timestamp)
        latest_accuracy = mlflow_client.get_run_metrics(latest_version.run_id).get(
            "test_accuracy"
        )

    total_versions = sum(
        len(mlflow_client.get_model_versions(model.name)) for model in models
    )

    return {
        "total_models": len(models),
        "total_versions": total_versions,
        "latest_model_name": latest_model_name,
        "latest_version": latest_version.version if latest_version is not None else None,
        "latest_stage": latest_stage,
        "latest_accuracy": latest_accuracy,
        "latest_created_time": latest_created_time,
    }


def _get_confusion_matrix(mlflow_client: MlflowRegistryClient, run_id: str):
    """
    The test-split confusion matrix for `run_id`, read from the
    metrics.json artifact MLflow tracking already logs — a confusion
    matrix isn't a scalar, so it was never logged as an MLflow metric,
    only inside this artifact. Returns None if the artifact is
    missing or unparseable rather than raising.
    """

    try:
        data = mlflow_client.get_artifact_bytes(run_id, "metrics.json")
        parsed = json.loads(data.decode("utf-8"))
        return parsed.get("test", {}).get("confusion_matrix")
    except Exception:
        return None


@router.get("/")
def home(
    request: Request,
    api_client: ApiClient = Depends(get_api_client),
    mlflow_client: MlflowRegistryClient = Depends(get_mlflow_client),
):
    health = None
    error = None

    try:
        health = api_client.get_health()
    except ApiUnavailableError as exc:
        error = str(exc)

    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "health": health,
            "error": error,
            "registry": _build_registry_summary(mlflow_client),
        },
    )


@router.get("/metrics")
def metrics_page(
    request: Request,
    mlflow_client: MlflowRegistryClient = Depends(get_mlflow_client),
):
    models = mlflow_client.get_registered_models()
    latest_model_name, latest_version = _latest_model_and_version(mlflow_client, models)

    metrics = None
    confusion_matrix = None
    if latest_version is not None:
        metrics = mlflow_client.get_run_metrics(latest_version.run_id)
        confusion_matrix = _get_confusion_matrix(mlflow_client, latest_version.run_id)

    return templates.TemplateResponse(
        request,
        "metrics.html",
        {
            "model_name": latest_model_name,
            "version": latest_version.version if latest_version is not None else None,
            "metrics": metrics,
            "confusion_matrix": confusion_matrix,
        },
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
def status_page(
    request: Request,
    api_client: ApiClient = Depends(get_api_client),
    mlflow_client: MlflowRegistryClient = Depends(get_mlflow_client),
    deployment_service: DeploymentService = Depends(get_deployment_service),
):
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

    mlflow_error = None
    try:
        registry = _build_registry_summary(mlflow_client)
        mlflow_ok = True
    except Exception as exc:
        mlflow_ok = False
        mlflow_error = str(exc)
        registry = {
            "total_models": 0,
            "total_versions": 0,
            "latest_created_time": None,
        }

    current_deployment = deployment_service.get_current_deployment()

    current_deployment_mlflow_version = None
    if mlflow_ok:
        try:
            current_deployment_mlflow_version = _find_mlflow_version_for_commit(
                mlflow_client, current_deployment.tag
            )
        except Exception:
            current_deployment_mlflow_version = None

    deployment_history = deployment_service.get_deployment_history()
    last_deployment = deployment_history[0] if deployment_history else None

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
            "mlflow_ok": mlflow_ok,
            "mlflow_error": mlflow_error,
            "registry": registry,
            "deployment_pipeline": get_deployment_pipeline_status(),
            "deployment_status": deployment_service.get_status(),
            "current_deployment": current_deployment,
            "current_deployment_mlflow_version": current_deployment_mlflow_version,
            "deployment_history": deployment_history,
            "last_deployment": last_deployment,
        },
    )


def _format_timestamp(epoch_millis: int) -> str:
    return datetime.fromtimestamp(epoch_millis / 1000, tz=timezone.utc).isoformat()


def _get_version_or_404(mlflow_client: MlflowRegistryClient, model_name: str, version: str):
    """Looks up one specific version of `model_name`, or 404s."""

    versions = mlflow_client.get_model_versions(model_name)
    matched = next((v for v in versions if str(v.version) == version), None)
    if matched is None:
        raise HTTPException(status_code=404, detail="Model version not found")
    return matched


def _find_mlflow_version_for_commit(mlflow_client: MlflowRegistryClient, git_commit_sha):
    """
    Reverse-looks-up which registered model version (if any) was
    trained from `git_commit_sha`, by scanning every version of every
    registered model for a matching git-commit tag. Never guessed —
    returns None if no version's tag matches, including when
    `git_commit_sha` itself is None (nothing currently deployed, or
    the deployed tag isn't a commit this registry knows about at all).

    O(models x versions) — deliberately unoptimized: this project's
    real scale (a handful of models, a few dozen versions) makes an
    index or cache unwarranted complexity for what's currently a
    Status-page-only, on-demand lookup.
    """

    if not git_commit_sha:
        return None

    for model in mlflow_client.get_registered_models():
        for version in mlflow_client.get_model_versions(model.name):
            if mlflow_client.get_git_commit(version.run_id) == git_commit_sha:
                return {"model_name": model.name, "version": version.version}
    return None


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


def _build_version_row(
    mlflow_client: MlflowRegistryClient,
    deployment_service: DeploymentService,
    commit_deployability_cache: dict,
    version,
) -> dict:
    metrics = mlflow_client.get_run_metrics(version.run_id)
    params = mlflow_client.get_run_parameters(version.run_id)

    git_commit = mlflow_client.get_git_commit(version.run_id)
    if git_commit not in commit_deployability_cache:
        # Memoized per page render only (not cached across requests):
        # several versions commonly share one commit (e.g. re-running
        # the pipeline without a new commit), so this avoids one real
        # GHCR query per version when they'd all resolve identically.
        commit_deployability_cache[git_commit] = (
            deployment_service.resolve_deployment_for_commit(git_commit)
        )
    deployability = commit_deployability_cache[git_commit]

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
        "git_commit": git_commit,
        "deployable": deployability.deployable,
        "deploy_reason": deployability.reason,
    }


@router.get("/models/{model_name}")
def model_detail(
    request: Request,
    model_name: str,
    mlflow_client: MlflowRegistryClient = Depends(get_mlflow_client),
    deployment_service: DeploymentService = Depends(get_deployment_service),
):
    versions = mlflow_client.get_model_versions(model_name)
    commit_deployability_cache = {}
    rows = [
        _build_version_row(mlflow_client, deployment_service, commit_deployability_cache, version)
        for version in versions
    ]

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
    matched = _get_version_or_404(mlflow_client, model_name, version)

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


def _deployment_action_confirm(
    request: Request,
    model_name: str,
    version: str,
    action: str,
    mlflow_client: MlflowRegistryClient,
    deployment_service: DeploymentService,
):
    """
    Shared GET handler for both Deploy and Rollback: mechanically
    identical (both resolve a version's git commit to a published
    image and, if confirmed, call deployment_service.deploy() with
    it) — only the confirmation copy differs, driven by `action`.
    """

    matched = _get_version_or_404(mlflow_client, model_name, version)
    git_commit = mlflow_client.get_git_commit(matched.run_id)
    check = deployment_service.resolve_deployment_for_commit(git_commit)

    if not check.deployable:
        return templates.TemplateResponse(
            request,
            "deployment_unavailable.html",
            {
                "action": action,
                "model_name": model_name,
                "version": version,
                "reason": check.reason,
            },
            status_code=409,
        )

    current = deployment_service.get_current_deployment()
    current_mlflow_version = _find_mlflow_version_for_commit(mlflow_client, current.tag)
    deployment_status = deployment_service.get_status()

    return templates.TemplateResponse(
        request,
        "deployment_confirm.html",
        {
            "action": action,
            "model_name": model_name,
            "version": version,
            "git_commit": git_commit,
            "image": check.image,
            "deployment_target": deployment_status.deployment_target,
            "current_image": current.image,
            "current_mlflow_version": current_mlflow_version,
        },
    )


def _deployment_action_execute(
    request: Request,
    model_name: str,
    version: str,
    action: str,
    mlflow_client: MlflowRegistryClient,
    deployment_service: DeploymentService,
):
    """
    Shared POST handler for both Deploy and Rollback. Re-verifies
    deployability rather than trusting the GET confirmation page's
    state — it's possible (if unlikely) for the published image to
    have disappeared between viewing the confirmation page and
    submitting it.
    """

    matched = _get_version_or_404(mlflow_client, model_name, version)
    git_commit = mlflow_client.get_git_commit(matched.run_id)
    check = deployment_service.resolve_deployment_for_commit(git_commit)

    if not check.deployable:
        return templates.TemplateResponse(
            request,
            "deployment_unavailable.html",
            {
                "action": action,
                "model_name": model_name,
                "version": version,
                "reason": check.reason,
            },
            status_code=409,
        )

    result = deployment_service.deploy(git_commit)

    return templates.TemplateResponse(
        request,
        "deployment_result.html",
        {
            "action": action,
            "model_name": model_name,
            "version": version,
            "result": result,
        },
    )


@router.get("/models/{model_name}/versions/{version}/actions/deploy")
def deploy_confirm(
    request: Request,
    model_name: str,
    version: str,
    mlflow_client: MlflowRegistryClient = Depends(get_mlflow_client),
    deployment_service: DeploymentService = Depends(get_deployment_service),
):
    return _deployment_action_confirm(
        request, model_name, version, "deploy", mlflow_client, deployment_service
    )


@router.post("/models/{model_name}/versions/{version}/actions/deploy")
def deploy_execute(
    request: Request,
    model_name: str,
    version: str,
    mlflow_client: MlflowRegistryClient = Depends(get_mlflow_client),
    deployment_service: DeploymentService = Depends(get_deployment_service),
):
    return _deployment_action_execute(
        request, model_name, version, "deploy", mlflow_client, deployment_service
    )


@router.get("/models/{model_name}/versions/{version}/actions/rollback")
def rollback_confirm(
    request: Request,
    model_name: str,
    version: str,
    mlflow_client: MlflowRegistryClient = Depends(get_mlflow_client),
    deployment_service: DeploymentService = Depends(get_deployment_service),
):
    return _deployment_action_confirm(
        request, model_name, version, "rollback", mlflow_client, deployment_service
    )


@router.post("/models/{model_name}/versions/{version}/actions/rollback")
def rollback_execute(
    request: Request,
    model_name: str,
    version: str,
    mlflow_client: MlflowRegistryClient = Depends(get_mlflow_client),
    deployment_service: DeploymentService = Depends(get_deployment_service),
):
    return _deployment_action_execute(
        request, model_name, version, "rollback", mlflow_client, deployment_service
    )


@router.get("/models/{model_name}/versions/{version}/actions/promote")
def promote_confirm(
    request: Request,
    model_name: str,
    version: str,
    mlflow_client: MlflowRegistryClient = Depends(get_mlflow_client),
):
    matched = _get_version_or_404(mlflow_client, model_name, version)

    return templates.TemplateResponse(
        request,
        "stage_transition_confirm.html",
        {
            "action": "promote",
            "model_name": model_name,
            "version": version,
            "current_stage": matched.current_stage,
        },
    )


@router.post("/models/{model_name}/versions/{version}/actions/promote")
def promote_execute(
    request: Request,
    model_name: str,
    version: str,
    mlflow_client: MlflowRegistryClient = Depends(get_mlflow_client),
):
    _get_version_or_404(mlflow_client, model_name, version)

    try:
        mlflow_client.promote_version(model_name, version)
        success, error = True, None
    except Exception as exc:
        success, error = False, str(exc)

    return templates.TemplateResponse(
        request,
        "stage_transition_result.html",
        {
            "action": "promote",
            "model_name": model_name,
            "version": version,
            "success": success,
            "error": error,
        },
    )


@router.get("/models/{model_name}/versions/{version}/actions/archive")
def archive_confirm(
    request: Request,
    model_name: str,
    version: str,
    mlflow_client: MlflowRegistryClient = Depends(get_mlflow_client),
):
    matched = _get_version_or_404(mlflow_client, model_name, version)

    return templates.TemplateResponse(
        request,
        "stage_transition_confirm.html",
        {
            "action": "archive",
            "model_name": model_name,
            "version": version,
            "current_stage": matched.current_stage,
        },
    )


@router.post("/models/{model_name}/versions/{version}/actions/archive")
def archive_execute(
    request: Request,
    model_name: str,
    version: str,
    mlflow_client: MlflowRegistryClient = Depends(get_mlflow_client),
):
    _get_version_or_404(mlflow_client, model_name, version)

    try:
        mlflow_client.archive_version(model_name, version)
        success, error = True, None
    except Exception as exc:
        success, error = False, str(exc)

    return templates.TemplateResponse(
        request,
        "stage_transition_result.html",
        {
            "action": "archive",
            "model_name": model_name,
            "version": version,
            "success": success,
            "error": error,
        },
    )
