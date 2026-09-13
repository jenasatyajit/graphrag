"""Qdrant wrapper tests with a fully mocked async client."""

from types import SimpleNamespace
from uuid import UUID

import pytest
from qdrant_client import models

from src.api.config import Settings
from src.api.errors import RetrievalError
from src.indexing.embeddings import EmbeddingResponse, SparseEmbedding
from src.indexing.vector_store import VectorStore
from src.ingestion.chunker import Chunk


class FakeQdrantClient:
    def __init__(self) -> None:
        self.created = False
        self.points = []
        self.closed = False

    async def collection_exists(self, name: str) -> bool:
        del name
        return False

    async def create_collection(self, **kwargs: object) -> None:
        self.created = True
        assert "vectors_config" in kwargs
        assert "sparse_vectors_config" in kwargs

    async def upsert(self, **kwargs: object) -> None:
        self.points = list(kwargs["points"])  # type: ignore[arg-type]

    async def query_points(self, **kwargs: object) -> SimpleNamespace:
        del kwargs
        payload = self.points[0].payload
        return SimpleNamespace(points=[SimpleNamespace(payload=payload, score=0.88)])

    async def get_collections(self) -> list[object]:
        return []

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_vector_store_upsert_and_dense_search_round_trip(settings: Settings) -> None:
    client = FakeQdrantClient()
    store = VectorStore(settings, client=client)
    chunk = Chunk(
        chunk_id=UUID("00000000-0000-0000-0000-000000000020"),
        doc_id=UUID("00000000-0000-0000-0000-000000000021"),
        doc_title="Security SOP",
        doc_type="SOP",
        text="Rotate credentials every 90 days.",
    )
    embeddings = EmbeddingResponse(
        dense=[[0.2] * 1024],
        sparse=[SparseEmbedding(indices=[5], values=[0.9])],
    )

    await store.connect()
    count = await store.upsert([chunk], embeddings)
    results = await store.dense_search([0.2] * 1024)
    await store.close()

    assert client.created is True
    assert count == 1
    assert results[0].chunk.chunk_id == chunk.chunk_id
    assert results[0].score == 0.88
    assert client.closed is True


class ExistingIncompatibleClient(FakeQdrantClient):
    async def collection_exists(self, name: str) -> bool:
        del name
        return True

    async def get_collection(self, name: str) -> SimpleNamespace:
        del name
        params = SimpleNamespace(
            vectors={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
            sparse_vectors={"sparse": models.SparseVectorParams()},
        )
        return SimpleNamespace(config=SimpleNamespace(params=params))


@pytest.mark.asyncio
async def test_existing_incompatible_collection_is_rejected(settings: Settings) -> None:
    store = VectorStore(settings, client=ExistingIncompatibleClient())

    with pytest.raises(RetrievalError, match="initialization failed"):
        await store.connect()
