"""
FastAPI application factory for the model serving layer.

All file I/O — reading the training contract, loading the model,
reading evaluation metrics — happens inside explicit function calls
made from create_app()/lifespan(), never as a side effect of merely
importing a module in this package. `app = create_app()` below is the
one standard exception: it's what lets `uvicorn app.main:app` find an
application instance to serve, but it's still an explicit call to a
documented entrypoint, not an accidental import-time read.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.contract import load_training_contract
from app.model_loader import load_model_bundle
from app.routes import build_router
from app.schemas import build_batch_request_model, build_prediction_request_model


def _build_lifespan(contract: dict):
    """
    Build a lifespan context manager bound to an already-loaded contract.

    Args:
        contract: The training contract, loaded once in create_app()
            and reused here so it isn't read from disk a second time.

    Returns:
        An async context manager suitable for FastAPI's `lifespan=`.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model_bundle = load_model_bundle(contract)
        yield

    return lifespan


def create_app() -> FastAPI:
    """
    Build a fully configured FastAPI application.

    Reading the training contract and constructing the dynamic
    prediction schemas both happen here, inside an explicit function
    call — not at import time. The actual model (the expensive,
    heavyweight resource) is loaded later, inside lifespan(), so a
    missing/corrupt model.pkl fails the app at startup rather than on
    the first request.

    Returns:
        A configured FastAPI application (routes registered, lifespan
        wired, but the model not yet loaded until the app starts).
    """

    contract = load_training_contract()

    prediction_model = build_prediction_request_model(contract)
    batch_model = build_batch_request_model(prediction_model)

    router = build_router(prediction_model, batch_model)

    app = FastAPI(
        title="MLOps Kubeflow Pipeline — Model Serving",
        lifespan=_build_lifespan(contract),
    )
    app.include_router(router)

    return app


app = create_app()
