"""Phase 7 degradation matrix: each dependency fails individually, end to end.

These tests drive the full stack through ``TestClient`` (not the router in
isolation) so that the HTTP status code, the uniform error envelope
``{error: {code, message, details, request_id}}``, and the ``X-Request-ID``
header are all exercised on every path. BACKEND.md Phase 7 T2 calls the
degradation matrix the headline deliverable: a dependency being down must
degrade quality to a valid, correctly-labeled response, never crash with a
leaked vendor stack trace.

The two true-failure paths (query embedding down; internal retrieval down AND
web empty) must return a clean 503 via the envelope, which the router documents
as the only cases that surface a dependency failure to the user.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from src.agent.router import WEB_UNAVAILABLE_NOTICE
from src.generation.synthesizer import CompletionResult
from src.retrieval.web_search import WebResult
from tests.conftest import (
    FakeEmbeddingProvider,
    FakeGraphStore,
    FakeReranker,
    FakeVectorStore,
    FakeWebSearch,
    ScriptedRouterLLM,
    build_client,
    tool_decision,
)

RETRIEVED_AT = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)


class ConstantCompleter:
    """Single reusable reply; create_app shares one provider across all calls."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def complete(self, *, system_prompt: str, user_prompt: str) -> CompletionResult:
        del system_prompt, user_prompt
        return CompletionResult(text=self._text)


def _web_result() -> WebResult:
    return WebResult(
        title="EU AI Act overview",
        url="https://example.org/ai-act",
        snippet="The EU AI Act entered into force in August 2024.",
        retrieved_at=RETRIEVED_AT,
    )


def _assert_error_envelope(payload: dict[str, Any], *, code: str) -> None:
    assert set(payload) == {"error"}
    error = payload["error"]
    assert set(error) == {"code", "message", "details", "request_id"}
    assert error["code"] == code
    # A clean, user-facing message, never a raw vendor/stack string.
    assert error["message"] and "Traceback" not in error["message"]
    assert error["request_id"]


# --- graceful degradation (still a valid 200 response) -----------------------


def test_reranker_failure_still_answers_from_rrf_order() -> None:
    # The reranker fails -> RRF order (no rerank_score). A chunk with no rerank
    # score counts as relevant, so the internal answer is still produced.
    with build_client(reranker=FakeReranker(fail=True)) as client:
        response = client.post("/query", json={"question": "Who approves large expenses?"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    payload = response.json()
    assert payload["answer"] == "Managers approve expenses above $500 [1]."
    assert payload["citations"][0]["source_tier"] == "internal"
    assert payload["used_web_search"] is False
    assert payload["notices"] == []


def test_graph_store_failure_degrades_to_vector_only_answer() -> None:
    # Graph store unavailable -> graph search degrades to no facts; the
    # vector-only answer is returned without a crash.
    with build_client(graph_store=FakeGraphStore(fail=True)) as client:
        response = client.post("/query", json={"question": "Who approves large expenses?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Managers approve expenses above $500 [1]."
    assert payload["citations"][0]["source_tier"] == "internal"


def test_web_search_failure_answers_internal_with_notice() -> None:
    # Web fails after an internal hit -> internal-only answer, the web-unavailable
    # notice is present, and used_web_search reflects reality (False).
    with build_client(
        router_llm=ScriptedRouterLLM(tool_decision("internal_search", "web_search")),
        web_searcher=FakeWebSearch(failed=True),
    ) as client:
        response = client.post(
            "/query",
            json={"question": "How do we approve expenses and what do regulators say?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["used_web_search"] is False
    assert payload["notices"] == [WEB_UNAVAILABLE_NOTICE]
    assert payload["citations"][0]["source_tier"] == "internal"


# --- true failures (clean 503 via the uniform envelope) ----------------------


def test_query_embedding_service_down_returns_clean_503_envelope() -> None:
    # The GPU VM embedding service is unreachable at query time: internal
    # retrieval raises RetrievalError. The router's guardrail escalates to web,
    # which is also unreachable here, so with no evidence anywhere the dependency
    # failure surfaces as a clean 503 envelope, not a raw stack trace.
    with build_client(
        embedding_provider=FakeEmbeddingProvider(fail=True),
        web_searcher=FakeWebSearch(failed=True),
    ) as client:
        response = client.post(
            "/query",
            json={"question": "Who approves large expenses?"},
            headers={"X-Request-ID": "embed-down-1"},
        )

    assert response.status_code == 503
    assert response.headers["X-Request-ID"] == "embed-down-1"
    payload = response.json()
    _assert_error_envelope(payload, code="retrieval_error")
    assert payload["error"]["request_id"] == "embed-down-1"


def test_internal_down_and_web_unavailable_returns_clean_503_envelope() -> None:
    # Internal retrieval is down (embedding unreachable) AND the web is also
    # unavailable: no evidence and no fallback anywhere -> the router surfaces
    # the dependency failure as a 503 envelope, never a fabricated answer. This
    # is the router's documented true-failure path.
    with build_client(
        embedding_provider=FakeEmbeddingProvider(fail=True),
        router_llm=ScriptedRouterLLM(tool_decision("internal_search", "web_search")),
        web_searcher=FakeWebSearch(failed=True),
    ) as client:
        response = client.post(
            "/query",
            json={"question": "What is the latest regulation on expenses?"},
            headers={"X-Request-ID": "total-outage-1"},
        )

    assert response.status_code == 503
    payload = response.json()
    _assert_error_envelope(payload, code="retrieval_error")
    assert payload["error"]["request_id"] == "total-outage-1"


def test_internal_down_but_web_reachable_empty_is_low_confidence_200() -> None:
    # The boundary case: internal is down but the web is reachable and simply
    # returns nothing. That is not a dependency failure (the web answered), so
    # the user gets a valid low-confidence 200 answer, not a 503.
    with build_client(
        embedding_provider=FakeEmbeddingProvider(fail=True),
        router_llm=ScriptedRouterLLM(tool_decision("internal_search", "web_search")),
        web_searcher=FakeWebSearch([]),  # reachable, zero results
    ) as client:
        response = client.post(
            "/query",
            json={"question": "What is the latest regulation on expenses?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"] == []
    assert payload["used_web_search"] is False


def test_internal_down_but_web_succeeds_still_answers_from_web() -> None:
    # The complement of the total-outage case: internal is down but the web
    # returns a result, so the request succeeds with an external-labeled answer
    # (degradation, not failure).
    with build_client(
        embedding_provider=FakeEmbeddingProvider(fail=True),
        completion_provider=ConstantCompleter("The Act is in force [1]."),
        router_llm=ScriptedRouterLLM(tool_decision("internal_search", "web_search")),
        web_searcher=FakeWebSearch([_web_result()]),
    ) as client:
        response = client.post(
            "/query",
            json={"question": "Is the EU AI Act in force?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["used_web_search"] is True
    assert payload["citations"][0]["source_tier"] == "external"
    assert payload["citations"][0]["reference"]["url"] == "https://example.org/ai-act"


def test_unexpected_vendor_error_never_leaks_as_stack_trace() -> None:
    # A non-domain exception from a dependency (here Qdrant) must be caught by
    # the catch-all handler and returned as the uniform internal_error envelope,
    # not as a raw 500 stack trace.
    class ExplodingVectorStore(FakeVectorStore):
        async def hybrid_search(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("qdrant: connection reset by peer")

    with build_client(vector_store=cast(Any, ExplodingVectorStore())) as client:
        response = client.post(
            "/query",
            json={"question": "Who approves large expenses?"},
            headers={"X-Request-ID": "boom-1"},
        )

    assert response.status_code == 500
    _assert_error_envelope(response.json(), code="internal_error")
    assert response.json()["error"]["request_id"] == "boom-1"
