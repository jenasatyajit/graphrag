"""Phase 2 hybrid retrieval (RRF) and OpenRouter rerank behavior tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import UUID

import httpx
import pytest
from qdrant_client import models

from src.api.config import Settings
from src.indexing.embeddings import SparseEmbedding
from src.indexing.vector_store import RetrievedChunk, VectorStore
from src.ingestion.chunker import Chunk
from src.retrieval.reranker import OpenRouterReranker


def _chunk(marker: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=UUID(f"00000000-0000-0000-0000-0000000000{marker}"),
        doc_id=UUID("00000000-0000-0000-0000-0000000000ff"),
        doc_title="Onboarding SOP",
        doc_type="SOP",
        text=text,
    )


def _retrieved(marker: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk=_chunk(marker, text), score=score)


class RecordingQdrantClient:
    """Fake async Qdrant client that records the query and returns fused points."""

    def __init__(self) -> None:
        self.query_kwargs: dict[str, Any] = {}

    async def query_points(self, **kwargs: Any) -> SimpleNamespace:
        self.query_kwargs = kwargs
        # Simulate the server-side RRF result: the chunk that ranked highly in
        # both branches surfaces first with the largest fused score.
        points = [
            SimpleNamespace(
                payload=_chunk("11", "reset the badge").model_dump(mode="json"), score=0.90
            ),
            SimpleNamespace(
                payload=_chunk("22", "escalate to IT").model_dump(mode="json"), score=0.40
            ),
        ]
        return SimpleNamespace(points=points)


@pytest.mark.asyncio
async def test_hybrid_search_issues_rrf_fusion_query(settings: Settings) -> None:
    client = RecordingQdrantClient()
    store = VectorStore(settings, client=client)

    results = await store.hybrid_search(
        [0.1] * 1024,
        SparseEmbedding(indices=[3, 9], values=[0.5, 0.2]),
        limit=20,
    )

    query = client.query_kwargs
    assert isinstance(query["query"], models.FusionQuery)
    assert query["query"].fusion == models.Fusion.RRF
    assert len(query["prefetch"]) == 2
    assert query["limit"] == 20
    # RRF fusion order is preserved and exposed as the retriever score.
    assert [str(r.chunk.chunk_id)[-2:] for r in results] == ["11", "22"]
    assert results[0].score == 0.90
    assert results[0].rerank_score is None


def _rerank_transport(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_reranker_reorders_and_keeps_scores(settings: Settings) -> None:
    candidates = [
        _retrieved("01", "escalate to IT", 0.30),
        _retrieved("02", "reset the badge", 0.20),
        _retrieved("03", "unrelated policy", 0.10),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.98},
                    {"index": 0, "relevance_score": 0.55},
                ]
            },
        )

    reranker = OpenRouterReranker(settings, client=_rerank_transport(handler))
    result = await reranker.rerank("how do I reset a badge?", candidates)
    await reranker.aclose()

    assert [str(r.chunk.chunk_id)[-2:] for r in result] == ["02", "01"]
    assert result[0].rerank_score == 0.98
    assert result[1].rerank_score == 0.55


@pytest.mark.asyncio
async def test_reranker_falls_back_to_rrf_order_on_failure(settings: Settings) -> None:
    candidates = [
        _retrieved("01", "first", 0.30),
        _retrieved("02", "second", 0.20),
        _retrieved("03", "third", 0.10),
        _retrieved("04", "fourth", 0.05),
        _retrieved("05", "fifth", 0.02),
        _retrieved("06", "sixth", 0.01),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "upstream unavailable"})

    reranker = OpenRouterReranker(settings, client=_rerank_transport(handler))
    result = await reranker.rerank("any question", candidates)
    await reranker.aclose()

    # No exception escapes; RRF order is preserved and truncated to top_n (5).
    assert len(result) == settings.rerank_top_n
    assert [str(r.chunk.chunk_id)[-2:] for r in result] == ["01", "02", "03", "04", "05"]
    assert all(r.rerank_score is None for r in result)
