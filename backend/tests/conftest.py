"""Shared deterministic fixtures for backend unit tests."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import timedelta
from typing import Any, Literal, TypedDict, cast
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from src.agent.llm import RouterDecision, ToolCall
from src.agent.memory import ConversationStore, InMemoryConversationStore
from src.api.config import Settings
from src.api.errors import RetrievalError
from src.api.main import create_app
from src.api.schemas import ComponentHealth
from src.generation.synthesizer import CompletionResult
from src.indexing.embeddings import EmbeddingMode, EmbeddingResponse, SparseEmbedding
from src.indexing.vector_store import RetrievedChunk
from src.ingestion.chunker import Chunk
from src.retrieval.web_search import WebResult, WebSearchOutcome


class TestSettingsValues(TypedDict):
    sarvam_api_key: str
    openrouter_api_key: str
    exa_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    app_environment: Literal["test"]


def settings_values() -> TestSettingsValues:
    return {
        "sarvam_api_key": "test-sarvam-key",
        "openrouter_api_key": "test-openrouter-key",
        "exa_api_key": "test-exa-key",
        "qdrant_url": "https://qdrant.example.test",
        "qdrant_api_key": "test-qdrant-key",
        "neo4j_uri": "neo4j+s://neo4j.example.test",
        "neo4j_user": "test-user",
        "neo4j_password": "test-password",
        "app_environment": "test",
    }


class FakeVectorStore:
    def __init__(self) -> None:
        self.source = RetrievedChunk(
            chunk=Chunk(
                chunk_id=UUID("00000000-0000-0000-0000-000000000001"),
                doc_id=UUID("00000000-0000-0000-0000-000000000002"),
                doc_title="Expense Approval SOP",
                doc_type="SOP",
                text="Managers approve expenses above $500.",
                page_number=2,
                sop_version="v2.1",
            ),
            score=0.91,
        )
        self.closed = False

    async def dense_search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        del query_vector, limit
        return [self.source]

    async def hybrid_search(
        self,
        query_dense: Sequence[float],
        query_sparse: object,
        *,
        limit: int | None = None,
        prefetch_limit: int | None = None,
    ) -> list[RetrievedChunk]:
        del query_dense, query_sparse, limit, prefetch_limit
        return [self.source]

    async def verify_connectivity(self) -> ComponentHealth:
        return ComponentHealth(name="qdrant", status="ok")

    async def close(self) -> None:
        self.closed = True


class FakeReranker:
    """Deterministic reranker. ``fail=True`` mimics the OpenRouter RRF fallback.

    The real ``OpenRouterReranker`` never raises: on any OpenRouter failure it
    returns the pre-rerank RRF order with ``rerank_score`` left as ``None``. The
    failing variant reproduces that contract so degradation tests exercise the
    same "no rerank score -> treated as relevant" path the live code takes.
    """

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        del query
        if self._fail:
            return [candidate.model_copy(update={"rerank_score": None}) for candidate in candidates]
        return list(candidates)


class FakeGraphStore:
    """Fake Neo4j store. ``fail=True`` raises from every query-time call.

    ``GraphSearchService`` swallows graph exceptions and degrades to no facts, so
    a failing store yields a vector-only answer rather than a crash.
    """

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail

    async def connect(self) -> None:
        return None

    async def verify_connectivity(self) -> ComponentHealth:
        if self._fail:
            return ComponentHealth(name="neo4j", status="unhealthy", detail="graph store down")
        return ComponentHealth(name="neo4j", status="ok")

    async def match_entities(self, names: Sequence[str]) -> list[str]:
        del names
        if self._fail:
            raise RuntimeError("neo4j unavailable")
        return []

    async def neighborhood_facts(self, node_names: Sequence[str]) -> list[Any]:
        del node_names
        if self._fail:
            raise RuntimeError("neo4j unavailable")
        return []

    async def close(self) -> None:
        return None


class FakeEmbeddingProvider:
    """Deterministic embedder. ``fail=True`` mimics an unreachable GPU VM.

    The real ``HttpEmbeddingClient`` raises ``RetrievalError`` (status 503 at the
    API boundary) when the embedding service is unreachable, so the failing
    variant raises the same domain error at query time.
    """

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail

    async def embed(
        self,
        texts: Sequence[str],
        *,
        mode: EmbeddingMode,
    ) -> EmbeddingResponse:
        del mode
        if self._fail:
            raise RetrievalError("The embedding service is temporarily unavailable.")
        return EmbeddingResponse(
            dense=[[0.1] * 1024 for _ in texts],
            sparse=[SparseEmbedding(indices=[7], values=[0.5]) for _ in texts],
        )

    async def verify_connectivity(self) -> ComponentHealth:
        if self._fail:
            return ComponentHealth(
                name="embedding_service", status="unhealthy", detail="embedding service down"
            )
        return ComponentHealth(name="embedding_service", status="ok")


class FakeCompletionProvider:
    async def complete(self, *, system_prompt: str, user_prompt: str) -> CompletionResult:
        del system_prompt, user_prompt
        return CompletionResult(text="Managers approve expenses above $500 [1].")


class ScriptedRouterLLM:
    """Fake tool-calling router returning scripted decisions, then DONE."""

    def __init__(self, *decisions: RouterDecision) -> None:
        self._decisions = list(decisions)
        self.calls: list[dict[str, Any]] = []

    async def decide(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]],
        tool_choice: str,
    ) -> RouterDecision:
        self.calls.append({"messages": list(messages), "tool_choice": tool_choice})
        if self._decisions:
            return self._decisions.pop(0)
        return RouterDecision(content="DONE")


def tool_decision(*names: str, query: str = "q") -> RouterDecision:
    return RouterDecision(
        tool_calls=[
            ToolCall(id=f"c{i}", name=name, arguments={"query": query})
            for i, name in enumerate(names)
        ]
    )


class FakeWebSearch:
    def __init__(self, results: list[WebResult] | None = None, *, failed: bool = False) -> None:
        self._results = results or []
        self._failed = failed
        self.queries: list[str] = []

    async def search(self, query: str) -> WebSearchOutcome:
        self.queries.append(query)
        return WebSearchOutcome(query=query, results=list(self._results), failed=self._failed)


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {**settings_values(), **overrides}
    return Settings(_env_file=None, **values)


def build_client(
    *,
    settings: Settings | None = None,
    vector_store: Any | None = None,
    graph_store: Any | None = None,
    embedding_provider: Any | None = None,
    completion_provider: Any | None = None,
    reranker: Any | None = None,
    router_llm: Any | None = None,
    web_searcher: Any | None = None,
    conversation_store: ConversationStore | None = None,
    raise_server_exceptions: bool = False,
) -> TestClient:
    """Build a ``TestClient`` over ``create_app`` with per-test fake overrides.

    Centralizes the full-stack wiring so degradation/e2e tests exercise the real
    error envelope, status codes, and ``X-Request-ID`` header end to end instead
    of calling the router in isolation. Defaults mirror the ``client`` fixture;
    pass failing variants of the fakes to drive a degradation path.
    """
    app = create_app(
        settings or make_settings(),
        vector_store=cast(Any, vector_store or FakeVectorStore()),
        graph_store=cast(Any, graph_store or FakeGraphStore()),
        embedding_provider=embedding_provider or FakeEmbeddingProvider(),
        completion_provider=completion_provider or FakeCompletionProvider(),
        reranker=reranker or FakeReranker(),
        router_llm=router_llm or ScriptedRouterLLM(tool_decision("internal_search")),
        web_searcher=web_searcher or FakeWebSearch(),
        conversation_store=conversation_store
        or InMemoryConversationStore(max_turns=20, ttl=timedelta(hours=1)),
    )
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def client(
    settings: Settings,
    fake_vector_store: FakeVectorStore,
) -> Iterator[TestClient]:
    app = create_app(
        settings,
        vector_store=cast(Any, fake_vector_store),
        graph_store=cast(Any, FakeGraphStore()),
        embedding_provider=FakeEmbeddingProvider(),
        completion_provider=FakeCompletionProvider(),
        reranker=FakeReranker(),
        router_llm=ScriptedRouterLLM(tool_decision("internal_search")),
        web_searcher=FakeWebSearch(),
        conversation_store=InMemoryConversationStore(max_turns=20, ttl=timedelta(hours=1)),
    )
    with TestClient(app) as test_client:
        yield test_client
