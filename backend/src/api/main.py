"""FastAPI application factory for the GraphRAG backend."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import Settings
from src.api.errors import register_exception_handlers
from src.api.logging import configure_logging
from src.api.middleware import RequestIdMiddleware
from src.api.routes import router
from src.indexing.graph_store import GraphStore
from src.indexing.vector_store import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure process-wide logging around the FastAPI lifecycle."""
    settings: Settings = app.state.settings
    configure_logging(settings.app_environment)
    logger = structlog.get_logger(__name__)
    logger.info("application_started", environment=settings.app_environment)
    try:
        yield
    finally:
        logger.info("application_stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the configured FastAPI application.

    Production startup should use ``uvicorn --factory src.api.main:create_app`` so
    configuration validation happens before the server begins accepting traffic.
    """
    active_settings = settings or Settings()
    app = FastAPI(
        title="Graph-Augmented Multimodal Company-Knowledge QA API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = active_settings
    app.state.vector_store = VectorStore(active_settings)
    app.state.graph_store = GraphStore(active_settings)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(router)
    register_exception_handlers(app)
    return app
