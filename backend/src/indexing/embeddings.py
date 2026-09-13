"""Shared BGE-M3 contract with GPU-local and HTTP embedding backends."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, Field, model_validator
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.api.errors import RetrievalError
from src.api.schemas import ComponentHealth

EmbeddingMode = Literal["query", "passage"]


class SparseEmbedding(BaseModel):
    """Qdrant-compatible sparse vector with deterministic index ordering."""

    indices: list[int]
    values: list[float]

    @model_validator(mode="after")
    def validate_sparse_vector(self) -> SparseEmbedding:
        if len(self.indices) != len(self.values):
            raise ValueError("sparse indices and values must have equal lengths")
        if any(index < 0 for index in self.indices):
            raise ValueError("sparse indices must be non-negative")
        if self.indices != sorted(set(self.indices)):
            raise ValueError("sparse indices must be unique and ascending")
        return self


class EmbeddingRequest(BaseModel):
    """Request accepted by every embedding backend."""

    texts: list[str] = Field(min_length=1)
    mode: EmbeddingMode = "passage"


class EmbeddingResponse(BaseModel):
    """Dense and sparse embeddings produced in the same model pass."""

    dense: list[list[float]]
    sparse: list[SparseEmbedding]

    @model_validator(mode="after")
    def validate_batch_lengths(self) -> EmbeddingResponse:
        if len(self.dense) != len(self.sparse):
            raise ValueError("dense and sparse batches must have equal lengths")
        if not self.dense:
            raise ValueError("embedding response cannot be empty")
        return self


class EmbeddingProvider(Protocol):
    """Common async interface used by ingestion and query-time retrieval."""

    async def embed(self, texts: Sequence[str], *, mode: EmbeddingMode) -> EmbeddingResponse: ...


class _BGEM3Model(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Mapping[str, Any]: ...


class InProcessBGEM3:
    """GPU-VM adapter around ``BGEM3FlagModel``.

    FlagEmbedding is imported lazily so importing the CPU API never imports
    torch or initializes CUDA.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda:0",
        use_fp16: bool = True,
        expected_dimension: int = 1024,
        inference_timeout_seconds: float = 90.0,
        model_factory: Callable[[], _BGEM3Model] | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._use_fp16 = use_fp16
        self._expected_dimension = expected_dimension
        self._inference_timeout_seconds = inference_timeout_seconds
        self._inference_lock = asyncio.Semaphore(1)
        self._model_factory = model_factory
        self._model: _BGEM3Model | None = None

    def load(self) -> None:
        """Load the model once, normally during embedding-service startup."""
        if self._model is not None:
            return
        if self._model_factory is not None:
            self._model = self._model_factory()
            return
        try:
            module = importlib.import_module("FlagEmbedding")
            model_class = module.BGEM3FlagModel
            self._model = model_class(
                self._model_name,
                use_fp16=self._use_fp16,
                devices=[self._device],
            )
        except Exception as exc:
            raise RetrievalError("The in-process embedding model could not be loaded.") from exc

    async def embed(
        self,
        texts: Sequence[str],
        *,
        mode: EmbeddingMode,
    ) -> EmbeddingResponse:
        """Return dense and lexical vectors for a non-empty text batch."""
        del mode  # BGE-M3 uses the same encode operation for queries/passages.
        clean_texts = _validate_texts(texts)
        self.load()
        assert self._model is not None
        try:
            async with self._inference_lock:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._model.encode,
                        clean_texts,
                        return_dense=True,
                        return_sparse=True,
                        return_colbert_vecs=False,
                    ),
                    timeout=self._inference_timeout_seconds,
                )
            response = _normalise_bge_output(result)
            _validate_response_cardinality(response, len(clean_texts))
            _validate_dense_dimensions(response, self._expected_dimension)
            return response
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError("The in-process embedding request failed.") from exc


class HttpEmbeddingClient:
    """Query-time client for the private GPU embedding service."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        expected_dimension: int = 1024,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._expected_dimension = expected_dimension
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    async def embed(
        self,
        texts: Sequence[str],
        *,
        mode: EmbeddingMode,
    ) -> EmbeddingResponse:
        request = EmbeddingRequest(texts=_validate_texts(texts), mode=mode)
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=0.25, min=0.25, max=2),
                retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
                reraise=True,
            ):
                with attempt:
                    response = await self._client.post(
                        f"{self._base_url}/embed",
                        headers=self._headers,
                        json=request.model_dump(mode="json"),
                        timeout=self._timeout,
                    )
                    if response.status_code == 429 or response.status_code >= 500:
                        raise httpx.TransportError(
                            f"Retryable embedding service status: {response.status_code}",
                            request=response.request,
                        )
                    response.raise_for_status()
            embedding = EmbeddingResponse.model_validate(response.json())
            _validate_response_cardinality(embedding, len(request.texts))
            _validate_dense_dimensions(embedding, self._expected_dimension)
            return embedding
        except Exception as exc:
            raise RetrievalError("The embedding service is temporarily unavailable.") from exc

    async def verify_connectivity(self) -> ComponentHealth:
        """Check service reachability without exposing endpoint credentials."""
        try:
            response = await self._client.get(
                f"{self._base_url}/health",
                headers=self._headers,
                timeout=self._timeout,
            )
            response.raise_for_status()
            return ComponentHealth(name="embedding_service", status="ok")
        except Exception:
            return ComponentHealth(
                name="embedding_service",
                status="unhealthy",
                detail="The query embedding service is unavailable.",
            )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _validate_texts(texts: Sequence[str]) -> list[str]:
    clean = [text.strip() for text in texts]
    if not clean or any(not text for text in clean):
        raise RetrievalError("Embedding input must contain non-empty text.")
    return clean


def _normalise_bge_output(result: Mapping[str, Any]) -> EmbeddingResponse:
    dense_raw = result.get("dense_vecs")
    sparse_raw = result.get("lexical_weights")
    if dense_raw is None or sparse_raw is None:
        raise RetrievalError("The embedding model returned an incomplete response.")

    dense_values = dense_raw.tolist() if hasattr(dense_raw, "tolist") else dense_raw
    dense = [[float(value) for value in vector] for vector in dense_values]
    sparse = [_normalise_sparse(weights) for weights in sparse_raw]
    return EmbeddingResponse(dense=dense, sparse=sparse)


def _normalise_sparse(weights: Mapping[Any, Any]) -> SparseEmbedding:
    pairs = sorted((int(index), float(value)) for index, value in weights.items())
    return SparseEmbedding(
        indices=[index for index, _ in pairs],
        values=[value for _, value in pairs],
    )


def _validate_response_cardinality(response: EmbeddingResponse, expected: int) -> None:
    if len(response.dense) != expected:
        raise RetrievalError("The embedding response count does not match the request.")


def _validate_dense_dimensions(response: EmbeddingResponse, expected: int) -> None:
    if any(len(vector) != expected for vector in response.dense):
        raise RetrievalError(
            "The embedding dimension does not match the configured Qdrant collection."
        )


async def close_embedding_provider(provider: EmbeddingProvider) -> None:
    """Close providers that own network resources."""
    close = getattr(provider, "aclose", None)
    if close is not None:
        result = close()
        if asyncio.iscoroutine(result):
            await result
