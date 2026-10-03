"""Exa wrapper tests (Exa is always faked; never called live here)."""

from __future__ import annotations

from datetime import UTC
from types import SimpleNamespace
from typing import Any

import pytest

from src.api.config import Settings
from src.generation.provenance import CitableSource, ground_answer
from src.retrieval.web_search import ExaWebSearch


class FakeExa:
    def __init__(self, response: Any = None, errors: list[Exception] | None = None) -> None:
        self._response = response
        self._errors = errors or []
        self.calls: list[dict[str, Any]] = []

    async def search(self, query: str, **kwargs: Any) -> Any:
        self.calls.append({"query": query, **kwargs})
        if self._errors:
            raise self._errors.pop(0)
        return self._response


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        results=[
            SimpleNamespace(
                url="https://example.org/a",
                title="Regulation update",
                highlights=["Rule A applies from 2026.", "Fines up to 4%."],
                score=0.8,
                published_date="2026-08-01",
            ),
            SimpleNamespace(url=None, title="no url", highlights=["dropped"]),
            SimpleNamespace(url="https://example.org/empty", title="empty", highlights=[]),
        ],
        cost_dollars=SimpleNamespace(total=0.005),
    )


@pytest.mark.asyncio
async def test_results_carry_url_and_utc_retrieved_at(settings: Settings) -> None:
    exa = FakeExa(_response())
    outcome = await ExaWebSearch(settings, client=exa).search("regulation update")

    assert outcome.failed is False
    assert [r.url for r in outcome.results] == ["https://example.org/a"]
    result = outcome.results[0]
    assert result.source_tier == "external"
    assert result.retrieved_at.tzinfo is UTC
    assert "Rule A" in result.snippet and "Fines" in result.snippet
    # Request shape: configured type/count, highlights nested under contents.
    call = exa.calls[0]
    assert call["type"] == settings.exa_search_type
    assert call["num_results"] == settings.exa_num_results
    assert call["contents"] == {
        "highlights": {"max_characters": settings.exa_highlight_max_characters}
    }


@pytest.mark.asyncio
async def test_exa_failure_degrades_without_raising(settings: Settings) -> None:
    exa = FakeExa(errors=[ValueError("Request failed with status code 401: bad key")])
    outcome = await ExaWebSearch(settings, client=exa).search("anything")

    assert outcome.failed is True
    assert outcome.results == []
    assert len(exa.calls) == 1  # 401 is not retried


@pytest.mark.asyncio
async def test_exa_transient_error_is_retried(settings: Settings) -> None:
    exa = FakeExa(_response(), errors=[ValueError("Request failed with status code 503: busy")])
    outcome = await ExaWebSearch(settings, client=exa).search("anything")

    assert outcome.failed is False
    assert len(exa.calls) == 2


@pytest.mark.asyncio
async def test_web_citation_is_external_with_url(settings: Settings) -> None:
    outcome = await ExaWebSearch(settings, client=FakeExa(_response())).search("q")
    grounded = ground_answer(
        "Rule A applies from 2026 [1].", [CitableSource.from_web(outcome.results[0])]
    )

    citation = grounded.citations[0]
    assert citation.source_tier == "external"
    assert citation.reference.url == "https://example.org/a"
    assert citation.reference.retrieved_at is not None
    assert citation.reference.is_current_version is None
    assert grounded.version_warning is None
