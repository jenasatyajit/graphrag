"""Dense+sparse embedding adapters and service tests."""

from collections.abc import Sequence

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.config import Settings
from src.api.errors import RetrievalError
from src.indexing.embedding_service import create_embedding_service_app
from src.indexing.embeddings import (
    EmbeddingMode,
    EmbeddingResponse,
    HttpEmbeddingClient,
    InProcessBGEM3,
    SparseEmbedding,
)
from tests.conftest import settings_values


class FakeModel:
    def encode(self, sentences: list[str], **kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "dense_vecs": [[0.25] * 1024 for _ in sentences],
            "lexical_weights": [{"9": 0.2, "2": 0.8} for _ in sentences],
        }


class FakeProvider:
    async def embed(
        self,
        texts: Sequence[str],
        *,
        mode: EmbeddingMode,
    ) -> EmbeddingResponse:
        del mode
        return EmbeddingResponse(
            dense=[[0.5] * 1024 for _ in texts],
            sparse=[SparseEmbedding(indices=[1], values=[0.7]) for _ in texts],
        )


@pytest.mark.asyncio
async def test_in_process_embedding_returns_dense_and_sorted_sparse() -> None:
    embedder = InProcessBGEM3("test", model_factory=FakeModel)

    result = await embedder.embed(["policy"], mode="passage")

    assert len(result.dense[0]) == 1024
    assert result.sparse[0].indices == [2, 9]
    assert result.sparse[0].values == [0.8, 0.2]


@pytest.mark.asyncio
async def test_http_embedding_client_validates_shared_contract() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embed"
        return httpx.Response(
            200,
            json={
                "dense": [[0.1] * 1024],
                "sparse": [{"indices": [3], "values": [0.4]}],
            },
        )

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    embedder = HttpEmbeddingClient("https://embed.example.test", client=async_client)

    result = await embedder.embed(["question"], mode="query")

    assert result.sparse[0].indices == [3]
    await async_client.aclose()


def test_embedding_service_enforces_token_and_returns_both_vectors() -> None:
    settings = Settings(
        _env_file=None,
        **settings_values(),
        embedding_service_token="private-token",
    )
    app = create_embedding_service_app(settings, provider=FakeProvider())

    with TestClient(app) as client:
        unauthorized = client.get("/health")
        response = client.post(
            "/embed",
            headers={"Authorization": "Bearer private-token"},
            json={"texts": ["hello"], "mode": "query"},
        )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert len(response.json()["dense"][0]) == 1024
    assert response.json()["sparse"][0]["indices"] == [1]


def test_embedding_service_requires_authentication_token() -> None:
    settings = Settings(_env_file=None, **settings_values())

    with pytest.raises(ValueError, match="EMBEDDING_SERVICE_TOKEN"):
        create_embedding_service_app(settings, provider=FakeProvider())


@pytest.mark.asyncio
async def test_http_embedding_client_retries_server_error() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(
            200,
            request=request,
            json={
                "dense": [[0.1] * 1024],
                "sparse": [{"indices": [3], "values": [0.4]}],
            },
        )

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    embedder = HttpEmbeddingClient("https://embed.example.test", client=async_client, max_retries=2)

    result = await embedder.embed(["question"], mode="query")

    assert len(result.dense) == 1
    assert attempts == 2
    await async_client.aclose()


@pytest.mark.asyncio
async def test_http_embedding_client_rejects_wrong_response_count() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "dense": [[0.1] * 1024],
                "sparse": [{"indices": [3], "values": [0.4]}],
            },
        )

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    embedder = HttpEmbeddingClient("https://embed.example.test", client=async_client)

    with pytest.raises(RetrievalError, match="temporarily unavailable"):
        await embedder.embed(["one", "two"], mode="query")
    await async_client.aclose()
