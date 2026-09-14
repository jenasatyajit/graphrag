"""Phase 1 dense-RAG orchestration seam, expanded into a router in Phase 5."""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from src.api.errors import NotFoundError, RetrievalError
from src.api.schemas import IngestionSummary
from src.generation.provenance import GroundedAnswer
from src.generation.synthesizer import AnswerSynthesizer
from src.indexing.embeddings import EmbeddingProvider
from src.indexing.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearchService
from src.retrieval.reranker import Reranker

logger = structlog.get_logger(__name__)


class NaiveRagService:
    """Run query embedding, dense retrieval, and grounded synthesis."""

    def __init__(
        self,
        embedder: EmbeddingProvider | None,
        vector_store: VectorStore,
        synthesizer: AnswerSynthesizer,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._synthesizer = synthesizer

    async def query(self, question: str) -> GroundedAnswer:
        if self._embedder is None:
            raise RetrievalError("The query embedding service is not configured.")
        embedding = await self._embedder.embed([question], mode="query")
        candidates = await self._vector_store.dense_search(embedding.dense[0])
        logger.info(
            "dense_retrieval_completed",
            candidate_count=len(candidates),
            chunk_ids=[str(candidate.chunk.chunk_id) for candidate in candidates],
        )
        return await self._synthesizer.synthesize(question, candidates)


class HybridRagService:
    """Phase 2 query path: hybrid retrieval, reranking, then grounded synthesis.

    ``NaiveRagService`` is retained as the Phase 6 ablation baseline; this
    service is the live query path once Phase 2 lands.
    """

    def __init__(
        self,
        hybrid_search: HybridSearchService,
        reranker: Reranker,
        synthesizer: AnswerSynthesizer,
    ) -> None:
        self._hybrid_search = hybrid_search
        self._reranker = reranker
        self._synthesizer = synthesizer

    async def query(self, question: str) -> GroundedAnswer:
        candidates = await self._hybrid_search.search(question)
        reranked = await self._reranker.rerank(question, candidates)
        logger.info(
            "hybrid_rerank_completed",
            candidate_count=len(candidates),
            reranked_count=len(reranked),
            chunk_ids=[str(candidate.chunk.chunk_id) for candidate in reranked],
        )
        return await self._synthesizer.synthesize(question, reranked)


class IngestionStatusService:
    """Read the latest GPU ingestion summary without starting a remote job."""

    def __init__(self, summary_path: Path) -> None:
        self._summary_path = summary_path

    def latest(self) -> IngestionSummary:
        try:
            payload = json.loads(self._summary_path.read_text(encoding="utf-8"))
            return IngestionSummary.model_validate(payload)
        except FileNotFoundError as exc:
            raise NotFoundError("No completed ingestion run was found.") from exc
        except (OSError, ValueError) as exc:
            raise NotFoundError("The latest ingestion summary is unavailable.") from exc
