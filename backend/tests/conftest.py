"""Shared deterministic fixtures for backend unit tests."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Literal, TypedDict, cast
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from src.api.config import Settings
from src.api.main import create_app
from src.api.schemas import ComponentHealth
from src.generation.synthesizer import CompletionResult
from src.indexing.embeddings import EmbeddingMode, EmbeddingResponse, SparseEmbedding
from src.indexing.vector_store import RetrievedChunk
from src.ingestion.chunker import Chunk


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
    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        del query
        return list(candidates)


class FakeGraphStore:
    async def connect(self) -> None:
        return None

    async def verify_connectivity(self) -> ComponentHealth:
        return ComponentHealth(name="neo4j", status="ok")

    async def match_entities(self, names: Sequence[str]) -> list[str]:
        del names
        return []

    async def neighborhood_facts(self, node_names: Sequence[str]) -> list[Any]:
        del node_names
        return []

    async def close(self) -> None:
        return None


class FakeEmbeddingProvider:
    async def embed(
        self,
        texts: Sequence[str],
        *,
        mode: EmbeddingMode,
    ) -> EmbeddingResponse:
        del mode
        return EmbeddingResponse(
            dense=[[0.1] * 1024 for _ in texts],
            sparse=[SparseEmbedding(indices=[7], values=[0.5]) for _ in texts],
        )

    async def verify_connectivity(self) -> ComponentHealth:
        return ComponentHealth(name="embedding_service", status="ok")


class FakeCompletionProvider:
    async def complete(self, *, system_prompt: str, user_prompt: str) -> CompletionResult:
        del system_prompt, user_prompt
        return CompletionResult(text="Managers approve expenses above $500 [1].")


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, **settings_values())


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
    )
    with TestClient(app) as test_client:
        yield test_client
