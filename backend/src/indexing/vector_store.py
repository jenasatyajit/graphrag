"""Async Qdrant wrapper for internal document chunks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from src.api.config import Settings
from src.api.errors import RetrievalError
from src.api.schemas import ComponentHealth
from src.indexing.embeddings import EmbeddingResponse, SparseEmbedding
from src.ingestion.chunker import Chunk


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval with fusion and optional rerank scores.

    ``score`` is the retriever score (dense similarity, or the RRF fusion score
    for hybrid search). ``rerank_score`` is the reranker relevance score when a
    reranker has processed the candidate; it stays ``None`` on the RRF-only
    fallback path so downstream logging can tell the two apart.
    """

    chunk: Chunk
    score: float
    rerank_score: float | None = None


class VectorStore:
    """Own Qdrant lifecycle, collection schema, indexing, and dense search."""

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client or AsyncQdrantClient(
            url=str(settings.qdrant_url),
            api_key=settings.qdrant_api_key.get_secret_value(),
            timeout=int(settings.embedding_request_timeout_seconds),
            check_compatibility=False,
        )

    async def connect(self) -> None:
        """Create the named dense+sparse collection when it does not exist."""
        try:
            exists = await self._client.collection_exists(self._settings.qdrant_collection_name)
            if exists:
                await self._validate_collection_schema()
                return
            await self._client.create_collection(
                collection_name=self._settings.qdrant_collection_name,
                vectors_config={
                    self._settings.qdrant_dense_vector_name: models.VectorParams(
                        size=self._settings.embedding_dimension,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    self._settings.qdrant_sparse_vector_name: models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                },
            )
        except Exception as exc:
            raise RetrievalError("Qdrant collection initialization failed.") from exc

    async def _validate_collection_schema(self) -> None:
        info = await self._client.get_collection(self._settings.qdrant_collection_name)
        parameters = info.config.params
        dense_vectors = parameters.vectors
        sparse_vectors = parameters.sparse_vectors or {}
        if not isinstance(dense_vectors, dict):
            raise RetrievalError("The Qdrant collection does not use named dense vectors.")
        dense = dense_vectors.get(self._settings.qdrant_dense_vector_name)
        if (
            dense is None
            or dense.size != self._settings.embedding_dimension
            or dense.distance != models.Distance.COSINE
            or self._settings.qdrant_sparse_vector_name not in sparse_vectors
        ):
            raise RetrievalError("The Qdrant collection schema is incompatible with this service.")

    async def upsert(self, chunks: Sequence[Chunk], embeddings: EmbeddingResponse) -> int:
        """Store chunks and both vector representations in one batch."""
        if len(chunks) != len(embeddings.dense):
            raise RetrievalError("Chunk and embedding batch lengths do not match.")
        points = [
            models.PointStruct(
                id=str(chunk.chunk_id),
                payload=chunk.model_dump(mode="json"),
                vector={
                    self._settings.qdrant_dense_vector_name: dense,
                    self._settings.qdrant_sparse_vector_name: models.SparseVector(
                        indices=sparse.indices,
                        values=sparse.values,
                    ),
                },
            )
            for chunk, dense, sparse in zip(
                chunks,
                embeddings.dense,
                embeddings.sparse,
                strict=True,
            )
        ]
        try:
            await self._client.upsert(
                collection_name=self._settings.qdrant_collection_name,
                points=points,
                wait=True,
            )
        except Exception as exc:
            raise RetrievalError("Qdrant could not index the document chunks.") from exc
        return len(points)

    async def dense_search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        """Search only the dense named vector for the Phase 1 baseline."""
        if len(query_vector) != self._settings.embedding_dimension:
            raise RetrievalError("Query embedding dimension is invalid.")
        try:
            response = await self._client.query_points(
                collection_name=self._settings.qdrant_collection_name,
                query=list(query_vector),
                using=self._settings.qdrant_dense_vector_name,
                limit=limit or self._settings.retrieval_top_k,
                with_payload=True,
            )
            return [
                RetrievedChunk(
                    chunk=Chunk.model_validate(point.payload),
                    score=float(point.score),
                )
                for point in response.points
                if point.payload is not None
            ]
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError("Qdrant dense retrieval failed.") from exc

    async def hybrid_search(
        self,
        query_dense: Sequence[float],
        query_sparse: SparseEmbedding,
        *,
        limit: int | None = None,
        prefetch_limit: int | None = None,
    ) -> list[RetrievedChunk]:
        """Fuse dense and sparse candidates with server-side RRF.

        Each modality is prefetched independently, then Qdrant merges the two
        ranked lists with Reciprocal Rank Fusion and returns the top ``limit``
        candidates. ``score`` on each result is the RRF fusion score.
        """
        if len(query_dense) != self._settings.embedding_dimension:
            raise RetrievalError("Query embedding dimension is invalid.")
        fused_limit = limit or self._settings.rerank_candidate_limit
        branch_limit = prefetch_limit or max(fused_limit * 2, fused_limit)
        try:
            response = await self._client.query_points(
                collection_name=self._settings.qdrant_collection_name,
                prefetch=[
                    models.Prefetch(
                        query=list(query_dense),
                        using=self._settings.qdrant_dense_vector_name,
                        limit=branch_limit,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=query_sparse.indices,
                            values=query_sparse.values,
                        ),
                        using=self._settings.qdrant_sparse_vector_name,
                        limit=branch_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=fused_limit,
                with_payload=True,
            )
            return [
                RetrievedChunk(
                    chunk=Chunk.model_validate(point.payload),
                    score=float(point.score),
                )
                for point in response.points
                if point.payload is not None
            ]
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError("Qdrant hybrid retrieval failed.") from exc

    async def verify_connectivity(self) -> ComponentHealth:
        """Return a safe health result for Qdrant Cloud."""
        try:
            exists = await self._client.collection_exists(self._settings.qdrant_collection_name)
            if not exists:
                return ComponentHealth(
                    name="qdrant",
                    status="unhealthy",
                    detail="The configured Qdrant collection does not exist.",
                )
            await self._validate_collection_schema()
            return ComponentHealth(name="qdrant", status="ok")
        except Exception:
            return ComponentHealth(
                name="qdrant",
                status="unhealthy",
                detail="Qdrant Cloud is unavailable or its collection schema is incompatible.",
            )

    async def close(self) -> None:
        """Close Qdrant network resources."""
        await self._client.close()
