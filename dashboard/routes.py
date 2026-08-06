"""
Dashboard page routes.

Every route calls the existing FastAPI service through ApiClient and
renders a template — no prediction/validation/training/evaluation
logic lives here, only presentation.
"""

from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates

from dashboard.api_client import ApiClient, ApiUnavailableError, get_api_client

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
