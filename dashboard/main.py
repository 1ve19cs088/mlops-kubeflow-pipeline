"""
FastAPI application factory for the dashboard.

A separate process that renders a web UI on top of the existing
serving API (app/), calling it over HTTP exactly as any other client
would. No business logic from app/ or src/ is imported or duplicated
here.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from dashboard.routes import router

DASHBOARD_DIR = Path(__file__).resolve().parent


def create_app() -> FastAPI:
    app = FastAPI(title="MLOps Dashboard")
    app.mount(
        "/static", StaticFiles(directory=str(DASHBOARD_DIR / "static")), name="static"
    )
    app.include_router(router)
    return app


app = create_app()
