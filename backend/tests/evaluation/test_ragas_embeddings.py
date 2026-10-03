"""Tests for the thin RAGAS embeddings adapter over HttpEmbeddingClient."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from src.evaluation.ragas_embeddings import HttpEmbeddingAdapter
from src.indexing.embeddings import EmbeddingMode, EmbeddingResponse, SparseEmbedding


class RecordingEmbeddingClient:
    """Fake embedder returning deterministic dense vectors and recording calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], EmbeddingMode]] = []

    async def embed(self, texts: Sequence[str], *, mode: EmbeddingMode) -> EmbeddingResponse:
        self.calls.append((list(texts), mode))
        dense = [[float(index), 0.5, 0.25] for index, _ in enumerate(texts)]
        sparse = [SparseEmbedding(indices=[0], values=[1.0]) for _ in texts]
        return EmbeddingResponse(dense=dense, sparse=sparse)


@pytest.mark.asyncio
async def test_aembed_text_returns_first_dense_vector() -> None:
    client = RecordingEmbeddingClient()
    adapter = HttpEmbeddingAdapter(client)

    vector = await adapter.aembed_text("hello")

    assert vector == [0.0, 0.5, 0.25]
    assert client.calls == [(["hello"], "query")]


@pytest.mark.asyncio
async def test_aembed_texts_batches_in_one_call() -> None:
    client = RecordingEmbeddingClient()
    adapter = HttpEmbeddingAdapter(client)

    vectors = await adapter.aembed_texts(["a", "b", "c"])

    assert vectors == [[0.0, 0.5, 0.25], [1.0, 0.5, 0.25], [2.0, 0.5, 0.25]]
    # A single batched service call, not one per text, in query mode.
    assert len(client.calls) == 1
    assert client.calls[0] == (["a", "b", "c"], "query")


@pytest.mark.asyncio
async def test_aembed_texts_empty_returns_empty_without_calling() -> None:
    client = RecordingEmbeddingClient()
    adapter = HttpEmbeddingAdapter(client)

    assert await adapter.aembed_texts([]) == []
    assert client.calls == []
