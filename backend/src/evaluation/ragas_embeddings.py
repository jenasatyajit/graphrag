"""Thin RAGAS embeddings adapter over the query-time HTTP embedding client.

``AnswerRelevancy`` needs an embeddings object to measure the cosine similarity
between the user's question and questions regenerated from the answer. Rather
than pulling a second embedding stack (torch/FlagEmbedding) into the CPU-only
eval closure, this adapter wraps the existing :class:`HttpEmbeddingClient` (the
BGE-M3 GPU service used at query time) and exposes exactly the modern RAGAS
``BaseRagasEmbedding`` interface. Only the dense vectors are returned, in
``query`` mode, matching how questions are embedded during retrieval.
"""

from __future__ import annotations

from typing import Any

from ragas.embeddings.base import BaseRagasEmbedding

from src.indexing.embeddings import EmbeddingProvider


class HttpEmbeddingAdapter(BaseRagasEmbedding):
    """Expose an :class:`EmbeddingProvider` through the RAGAS embedding ABC.

    ``AnswerRelevancy`` only calls the async methods (``aembed_text`` /
    ``aembed_texts``); the sync ``embed_text`` is required by the ABC but is not
    used on the metric's hot path, so it defers to the shared async-in-loop
    helper provided by the base class.
    """

    def __init__(self, embedder: EmbeddingProvider) -> None:
        super().__init__()
        self._embedder = embedder

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        """Synchronous single-text embedding (ABC requirement; async path preferred)."""
        del kwargs
        return self._run_async_in_current_loop(self.aembed_text(text))

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        """Return the dense query embedding for a single text."""
        del kwargs
        response = await self._embedder.embed([text], mode="query")
        return list(response.dense[0])

    async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Batch-embed in a single service call, overriding the per-text default."""
        del kwargs
        if not texts:
            return []
        response = await self._embedder.embed(texts, mode="query")
        return [list(vector) for vector in response.dense]
