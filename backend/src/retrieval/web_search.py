"""Per-request Exa live web search with external provenance (SPEC 7.3).

Every result carries its source ``url`` and a UTC ``retrieved_at`` timestamp so
it can be cited as ``source_tier="external"``. Results are returned to the
caller only: nothing here writes to Qdrant or Neo4j, so web content can never
leak into the internal chunk store.

Exa failures never raise out of :meth:`ExaWebSearch.search`; they return a
:class:`WebSearchOutcome` with ``failed=True`` so the router can answer from
internal sources and tell the user web search was unavailable.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx
import structlog
from pydantic import BaseModel, Field
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from src.api.config import Settings

logger = structlog.get_logger(__name__)

_STATUS_RE = re.compile(r"status code (\d{3})")


class WebResult(BaseModel):
    """One external search hit, held only in the current request."""

    title: str
    url: str
    snippet: str
    published_date: str | None = None
    score: float | None = None
    retrieved_at: datetime
    source_tier: str = Field(default="external", frozen=True)


class WebSearchOutcome(BaseModel):
    """Results of one web search call, plus whether the call failed."""

    query: str
    results: list[WebResult] = Field(default_factory=list)
    failed: bool = False


class WebSearcher(Protocol):
    async def search(self, query: str) -> WebSearchOutcome: ...


class ExaWebSearch:
    """Thin resilient boundary around the async Exa SDK."""

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self._search_type = settings.exa_search_type
        self._num_results = settings.exa_num_results
        self._max_chars = settings.exa_highlight_max_characters
        self._timeout = settings.exa_request_timeout_seconds
        self._max_retries = settings.exa_max_retries
        if client is None:
            from exa_py import AsyncExa

            client = AsyncExa(api_key=settings.exa_api_key.get_secret_value())
        self._client = client

    async def search(self, query: str) -> WebSearchOutcome:
        """Return external results for ``query``; never raises on Exa failure."""
        cleaned = query.strip()
        if not cleaned:
            return WebSearchOutcome(query=query, failed=True)
        try:
            response = await self._call_exa(cleaned)
        except Exception:
            logger.warning("web_search_failed", query_length=len(cleaned), exc_info=True)
            return WebSearchOutcome(query=cleaned, failed=True)

        retrieved_at = datetime.now(UTC)
        results = [
            result
            for item in getattr(response, "results", None) or []
            if (result := _to_web_result(item, retrieved_at, self._max_chars)) is not None
        ]
        logger.info(
            "web_search_completed",
            query_length=len(cleaned),
            result_count=len(results),
            urls=[result.url for result in results],
            cost_dollars=_cost(response),
        )
        return WebSearchOutcome(query=cleaned, results=results)

    async def _call_exa(self, query: str) -> Any:
        response: Any = None
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception(_is_retryable_exa_error),
            reraise=True,
        ):
            with attempt:
                # The SDK's own httpx client uses a 600s timeout, so bound it here.
                response = await asyncio.wait_for(
                    self._client.search(
                        query,
                        type=self._search_type,
                        num_results=self._num_results,
                        contents={"highlights": {"max_characters": self._max_chars}},
                    ),
                    timeout=self._timeout,
                )
        return response

    async def aclose(self) -> None:
        inner = getattr(self._client, "_client", None)
        close = getattr(inner, "aclose", None)
        if close is not None:
            await close()


def _is_retryable_exa_error(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TransportError, TimeoutError)):
        return True
    if isinstance(exc, ValueError):
        match = _STATUS_RE.search(str(exc))
        if match:
            status = int(match.group(1))
            return status == 429 or status >= 500
    return False


def _to_web_result(item: Any, retrieved_at: datetime, max_chars: int) -> WebResult | None:
    url = getattr(item, "url", None)
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return None
    highlights = getattr(item, "highlights", None) or []
    snippet = " ... ".join(h.strip() for h in highlights if isinstance(h, str) and h.strip())
    if not snippet:
        text = getattr(item, "text", None)
        snippet = text.strip() if isinstance(text, str) else ""
    if not snippet:
        return None
    title = getattr(item, "title", None)
    score = getattr(item, "score", None)
    published = getattr(item, "published_date", None)
    return WebResult(
        title=title.strip() if isinstance(title, str) and title.strip() else url,
        url=url,
        snippet=snippet[:max_chars],
        published_date=published if isinstance(published, str) else None,
        score=float(score) if isinstance(score, (int, float)) else None,
        retrieved_at=retrieved_at,
    )


def _cost(response: Any) -> float | None:
    cost = getattr(response, "cost_dollars", None)
    total = getattr(cost, "total", None)
    return float(total) if isinstance(total, (int, float)) else None
