"""Resilient OpenRouter reranking with graceful RRF fallback.

Reranking runs on every query and sits in the user-facing latency path, so a
reranker outage must never break retrieval (SPEC 7.3, 13). On any failure the
reranker returns the pre-rerank RRF order truncated to ``top_n`` instead of
raising. Rerank goes through OpenRouter's OpenAI-style ``/rerank`` endpoint
rather than Cohere's native SDK (see BACKEND.md deviation log).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import httpx
import structlog
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.api.config import Settings
from src.indexing.vector_store import RetrievedChunk

logger = structlog.get_logger(__name__)


class Reranker(Protocol):
    """Reorder retrieved candidates by query relevance."""

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk]
    ) -> list[RetrievedChunk]: ...


class OpenRouterReranker:
    """Batch-rerank candidates via OpenRouter, degrading to RRF order on error."""

    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        self._model = settings.rerank_model
        self._top_n = settings.rerank_top_n
        self._timeout = settings.rerank_request_timeout_seconds
        self._max_retries = settings.rerank_max_retries
        self._endpoint = f"{str(settings.openrouter_base_url).rstrip('/')}/rerank"
        self._headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}"
        }
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Return the top-n reranked candidates, or RRF order if rerank fails."""
        ordered = list(candidates)
        if not ordered:
            return []

        fallback = ordered[: self._top_n]
        try:
            payload = await self._call_openrouter(query, ordered)
            reranked = self._merge_scores(payload, ordered)
        except Exception:
            logger.warning(
                "rerank_fallback_to_rrf",
                model=self._model,
                candidate_count=len(ordered),
                exc_info=True,
            )
            return fallback

        if not reranked:
            return fallback
        logger.info(
            "rerank_completed",
            model=self._model,
            candidate_count=len(ordered),
            result_count=len(reranked),
            rerank_scores=[round(item.rerank_score or 0.0, 4) for item in reranked],
        )
        return reranked

    async def _call_openrouter(
        self, query: str, candidates: Sequence[RetrievedChunk]
    ) -> dict[str, Any]:
        body = {
            "model": self._model,
            "query": query,
            "documents": [candidate.chunk.text for candidate in candidates],
            "top_n": min(self._top_n, len(candidates)),
        }
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=0.25, min=0.25, max=2),
            retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
            reraise=True,
        ):
            with attempt:
                response = await self._client.post(
                    self._endpoint,
                    headers=self._headers,
                    json=body,
                    timeout=self._timeout,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.TransportError(
                        f"Retryable rerank status: {response.status_code}",
                        request=response.request,
                    )
                response.raise_for_status()
        return response.json()

    def _merge_scores(
        self, payload: dict[str, Any], candidates: Sequence[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            raise ValueError("OpenRouter rerank response contained no results.")

        reranked: list[RetrievedChunk] = []
        for item in results:
            index = item.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(candidates):
                raise ValueError("OpenRouter rerank returned an out-of-range index.")
            score = item.get("relevance_score")
            source = candidates[index]
            reranked.append(
                source.model_copy(
                    update={"rerank_score": float(score) if score is not None else None}
                )
            )
        return reranked

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
