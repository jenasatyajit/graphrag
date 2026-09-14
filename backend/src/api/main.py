"""FastAPI application factory for the GraphRAG backend."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.agent.router import HybridRagService, IngestionStatusService
from src.api.config import Settings
from src.api.errors import RetrievalError, register_exception_handlers
from src.api.logging import configure_logging
from src.api.middleware import RequestIdMiddleware
from src.api.routes import router
from src.api.schemas import ComponentHealth
from src.generation.synthesizer import AnswerSynthesizer, SarvamCompletionProvider
from src.indexing.embeddings import HttpEmbeddingClient, close_embedding_provider
from src.indexing.graph_store import GraphStore
from src.indexing.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearchService
from src.retrieval.reranker import OpenRouterReranker, Reranker


class MissingEmbeddingService:
    """Health-only adapter used when the CPU API lacks a service URL."""

    async def verify_connectivity(self) -> ComponentHealth:
        return ComponentHealth(
            name="embedding_service",
            status="unhealthy",
            detail="EMBEDDING_SERVICE_URL is not configured.",
        )


class UnconfiguredQueryService:
    """Query service used when no embedding provider is configured."""

    async def query(self, question: str) -> None:
        del question
        raise RetrievalError("The query embedding service is not configured.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging and close owned external clients on shutdown."""
    settings: Settings = app.state.settings
    configure_logging(settings.app_environment)
    logger = structlog.get_logger(__name__)
    logger.info("application_started", environment=settings.app_environment)
    try:
        yield
    finally:
        await app.state.vector_store.close()
        embedder = app.state.embedding_provider
        if embedder is not None:
            await close_embedding_provider(embedder)
        reranker = app.state.reranker
        close_reranker = getattr(reranker, "aclose", None)
        if close_reranker is not None:
            await close_reranker()
        logger.info("application_stopped")


def create_app(
    settings: Settings | None = None,
    *,
    vector_store: VectorStore | None = None,
    graph_store: GraphStore | None = None,
    embedding_provider: Any | None = None,
    completion_provider: Any | None = None,
    reranker: Reranker | None = None,
) -> FastAPI:
    """Create the configured CPU-only query API application."""
    active_settings = settings or Settings()
    active_vector_store = vector_store or VectorStore(active_settings)
    active_graph_store = graph_store or GraphStore(active_settings)

    if embedding_provider is None and active_settings.embedding_service_url is not None:
        token = active_settings.embedding_service_token
        embedding_provider = HttpEmbeddingClient(
            str(active_settings.embedding_service_url),
            token=token.get_secret_value() if token is not None else None,
            timeout_seconds=active_settings.embedding_request_timeout_seconds,
            max_retries=active_settings.embedding_max_retries,
            expected_dimension=active_settings.embedding_dimension,
        )
    embedding_health = embedding_provider or MissingEmbeddingService()
    active_reranker = reranker or OpenRouterReranker(active_settings)
    active_completion_provider = completion_provider or SarvamCompletionProvider(active_settings)
    synthesizer = AnswerSynthesizer(active_completion_provider)

    app = FastAPI(
        title="Graph-Augmented Multimodal Company-Knowledge QA API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = active_settings
    app.state.vector_store = active_vector_store
    app.state.graph_store = active_graph_store
    app.state.embedding_provider = embedding_provider
    app.state.embedding_health = embedding_health
    app.state.reranker = active_reranker
    if embedding_provider is None:
        app.state.query_service = UnconfiguredQueryService()
    else:
        hybrid_search = HybridSearchService(embedding_provider, active_vector_store)
        app.state.query_service = HybridRagService(
            hybrid_search,
            active_reranker,
            synthesizer,
        )
    app.state.ingestion_status = IngestionStatusService(active_settings.ingestion_summary_path)

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
