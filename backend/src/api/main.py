"""FastAPI application factory for the GraphRAG backend."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.agent.llm import SarvamToolCallingProvider, ToolCallingProvider
from src.agent.memory import ConversationStore, FollowUpRewriter, build_conversation_store
from src.agent.router import AgentRouter, HybridRagService, IngestionStatusService
from src.api.config import Settings
from src.api.errors import RetrievalError, register_exception_handlers
from src.api.jobs import JobStore
from src.api.logging import configure_logging
from src.api.middleware import RequestIdMiddleware
from src.api.routes import router
from src.api.schemas import ComponentHealth
from src.generation.synthesizer import AnswerSynthesizer, SarvamCompletionProvider
from src.indexing.embeddings import HttpEmbeddingClient, close_embedding_provider
from src.indexing.graph_store import GraphStore
from src.indexing.vector_store import VectorStore
from src.retrieval.graph_search import GraphSearchService
from src.retrieval.hybrid_search import HybridSearchService
from src.retrieval.reranker import OpenRouterReranker, Reranker
from src.retrieval.web_search import ExaWebSearch, WebSearcher


class MissingEmbeddingService:
    """Health-only adapter used when the CPU API lacks a service URL."""

    async def verify_connectivity(self) -> ComponentHealth:
        return ComponentHealth(
            name="embedding_service",
            status="unhealthy",
            detail="EMBEDDING_SERVICE_URL is not configured.",
        )


class UnconfiguredAgent:
    """Agent used when no embedding provider is configured."""

    async def answer(self, question: str, conversation_id: str | None = None) -> None:
        del question, conversation_id
        raise RetrievalError("The query embedding service is not configured.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging and close owned external clients on shutdown."""
    settings: Settings = app.state.settings
    configure_logging(settings.app_environment)
    logger = structlog.get_logger(__name__)
    logger.info("application_started", environment=settings.app_environment)
    try:
        # Best-effort graph connect: a graph outage must not stop the app from
        # serving vector-only answers, so failures are logged, not raised.
        try:
            await app.state.graph_store.connect()
        except Exception:
            logger.warning("graph_store_connect_failed", exc_info=True)
        yield
    finally:
        await app.state.vector_store.close()
        await app.state.graph_store.close()
        embedder = app.state.embedding_provider
        if embedder is not None:
            await close_embedding_provider(embedder)
        for owned in (app.state.reranker, app.state.web_searcher):
            close_owned = getattr(owned, "aclose", None)
            if close_owned is not None:
                await close_owned()
        await app.state.conversation_store.close()
        logger.info("application_stopped")


def create_app(
    settings: Settings | None = None,
    *,
    vector_store: VectorStore | None = None,
    graph_store: GraphStore | None = None,
    embedding_provider: Any | None = None,
    completion_provider: Any | None = None,
    reranker: Reranker | None = None,
    router_llm: ToolCallingProvider | None = None,
    web_searcher: WebSearcher | None = None,
    conversation_store: ConversationStore | None = None,
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
    entity_extractor = completion_provider or SarvamCompletionProvider(
        active_settings,
        reasoning_effort=active_settings.llm_extract_reasoning_effort,
        max_tokens=active_settings.llm_extract_max_tokens,
    )
    graph_search = GraphSearchService(active_graph_store, entity_extractor)
    # Router-adjacent calls (follow-up rewrite, direct answers) are cheap
    # classification-style calls: reasoning disabled, small token budget.
    fast_completion = completion_provider or SarvamCompletionProvider(
        active_settings,
        reasoning_effort=active_settings.llm_router_reasoning_effort,
        max_tokens=active_settings.llm_router_max_tokens,
    )
    active_router_llm = router_llm or SarvamToolCallingProvider(active_settings)
    active_web = web_searcher or ExaWebSearch(active_settings)
    active_store = conversation_store or build_conversation_store(active_settings)

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
    app.state.web_searcher = active_web
    app.state.conversation_store = active_store
    if embedding_provider is None:
        app.state.agent = UnconfiguredAgent()
    else:
        hybrid_search = HybridSearchService(embedding_provider, active_vector_store)
        internal = HybridRagService(hybrid_search, active_reranker, synthesizer, graph_search)
        app.state.agent = AgentRouter(
            router_llm=active_router_llm,
            internal=internal,
            web=active_web,
            synthesizer=synthesizer,
            direct_responder=fast_completion,
            conversation_store=active_store,
            rewriter=FollowUpRewriter(fast_completion),
            max_iterations=active_settings.max_agent_iterations,
            relevance_threshold=active_settings.internal_relevance_threshold,
            context_turns=active_settings.conversation_context_turns,
            max_internal_sources=active_settings.rerank_top_n * 2,
        )
    app.state.ingestion_status = IngestionStatusService(active_settings.ingestion_summary_path)
    # In-process ingestion job registry (jobs are lost on restart; see JobStore).
    app.state.job_store = JobStore()

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
