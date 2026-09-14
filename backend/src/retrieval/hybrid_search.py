"""Dense + sparse retrieval fused with reciprocal-rank fusion in Qdrant."""

from __future__ import annotations

import structlog

from src.indexing.embeddings import EmbeddingProvider
from src.indexing.vector_store import RetrievedChunk, VectorStore

logger = structlog.get_logger(__name__)


class HybridSearchService:
    """Embed a question once and retrieve RRF-fused hybrid candidates.

    BGE-M3 produces the dense and sparse vectors in a single pass, so this
    service embeds the query in ``query`` mode and hands both representations to
    the vector store, which performs the Reciprocal Rank Fusion server-side.
    """

    def __init__(self, embedder: EmbeddingProvider, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    async def search(
        self,
        question: str,
        *,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return fused hybrid candidates for a single user question."""
        embedding = await self._embedder.embed([question], mode="query")
        candidates = await self._vector_store.hybrid_search(
            embedding.dense[0],
            embedding.sparse[0],
            limit=limit,
        )
        logger.info(
            "hybrid_retrieval_completed",
            candidate_count=len(candidates),
            chunk_ids=[str(candidate.chunk.chunk_id) for candidate in candidates],
        )
        return candidates
