"""Router agent tests: routing rules, guardrails, cap, provenance, and memory.

All vendors are faked: the tool-calling LLM is scripted, internal retrieval is
a stub, Exa is a fake, and synthesis runs through the real AnswerSynthesizer +
grounding with a scripted completion provider.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from src.agent.llm import RouterDecision, ToolCall
from src.agent.memory import ConversationTurn, FollowUpRewriter, InMemoryConversationStore
from src.agent.prompts import DIRECT_ANSWER_SENTINEL
from src.agent.router import (
    LOW_CONFIDENCE_ANSWER,
    WEB_UNAVAILABLE_NOTICE,
    AgentRouter,
    HybridRagService,
    NaiveRagService,
)
from src.api.config import Settings
from src.api.errors import LLMError, RetrievalError
from src.api.main import create_app
from src.generation.synthesizer import AnswerSynthesizer, CompletionResult
from src.indexing.graph_schema import GraphFact
from src.indexing.vector_store import RetrievedChunk
from src.ingestion.chunker import Chunk
from src.retrieval.web_search import WebResult
from tests.conftest import (
    FakeCompletionProvider,
    FakeEmbeddingProvider,
    FakeGraphStore,
    FakeReranker,
    FakeVectorStore,
    FakeWebSearch,
    ScriptedRouterLLM,
    tool_decision,
)

RETRIEVED_AT = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)


# --- fakes -------------------------------------------------------------------


def _chunk(
    text: str = "Managers approve expenses above $500.",
    *,
    rerank: float | None = 0.9,
    current: bool = True,
    version: str = "v3.0",
    marker: str = "01",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=UUID(f"00000000-0000-0000-0000-0000000000{marker}"),
            doc_id=UUID("00000000-0000-0000-0000-0000000000aa"),
            doc_title="Expense Approval SOP",
            doc_type="SOP",
            text=text,
            sop_version=version,
            is_current_version=current,
        ),
        score=0.03,
        rerank_score=rerank,
    )


def _web_result() -> WebResult:
    return WebResult(
        title="EU AI Act overview",
        url="https://example.org/ai-act",
        snippet="The EU AI Act entered into force in August 2024.",
        retrieved_at=RETRIEVED_AT,
    )


class StubInternal:
    def __init__(
        self,
        chunks: list[RetrievedChunk] | None = None,
        facts: list[GraphFact] | None = None,
    ) -> None:
        self._chunks = chunks or []
        self._facts = facts or []
        self.queries: list[str] = []

    async def retrieve(self, question: str) -> tuple[list[RetrievedChunk], list[GraphFact]]:
        self.queries.append(question)
        return list(self._chunks), list(self._facts)


class ScriptedCompleter:
    """Completion fake that replays replies and records prompts."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    async def complete(self, *, system_prompt: str, user_prompt: str) -> CompletionResult:
        self.prompts.append((system_prompt, user_prompt))
        reply = self._replies.pop(0) if self._replies else "Insufficient context to answer."
        return CompletionResult(text=reply)


class FailingRouterLLM:
    async def decide(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]],
        tool_choice: str,
    ) -> RouterDecision:
        raise LLMError("router down")


def _router(
    *,
    llm: Any,
    internal: StubInternal | None = None,
    web: FakeWebSearch | None = None,
    synth: ScriptedCompleter | None = None,
    direct: ScriptedCompleter | None = None,
    rewriter: ScriptedCompleter | None = None,
    store: InMemoryConversationStore | None = None,
    max_iterations: int = 3,
) -> AgentRouter:
    return AgentRouter(
        router_llm=llm,
        internal=cast(HybridRagService, internal or StubInternal([_chunk()])),
        web=web or FakeWebSearch(),
        synthesizer=AnswerSynthesizer(synth or ScriptedCompleter("Managers approve [1].")),
        direct_responder=direct or ScriptedCompleter("Hello!"),
        conversation_store=store or InMemoryConversationStore(max_turns=20, ttl=timedelta(hours=1)),
        rewriter=FollowUpRewriter(rewriter or ScriptedCompleter()),
        max_iterations=max_iterations,
        relevance_threshold=0.2,
        context_turns=3,
        max_internal_sources=10,
    )


# --- baseline kept for the ablation study ---------------------------------------


@pytest.mark.asyncio
async def test_naive_rag_service_runs_embed_retrieve_synthesize() -> None:
    service = NaiveRagService(
        FakeEmbeddingProvider(),
        FakeVectorStore(),  # type: ignore[arg-type]
        AnswerSynthesizer(FakeCompletionProvider()),
    )

    result = await service.query("Who approves large expenses?")

    assert result.answer == "Managers approve expenses above $500 [1]."
    assert len(result.citations) == 1


# --- routing rules --------------------------------------------------------------


@pytest.mark.asyncio
async def test_sop_question_routes_to_internal_search() -> None:
    llm = ScriptedRouterLLM(tool_decision("internal_search", query="expense approval"))
    internal = StubInternal([_chunk()])
    web = FakeWebSearch([_web_result()])
    router = _router(llm=llm, internal=internal, web=web)

    result = await router.answer("Who approves expenses over $500?")

    assert result.tool_trace == ["internal_search"]
    assert result.used_web_search is False
    assert internal.queries == ["expense approval"]
    assert web.queries == []
    assert llm.calls[0]["tool_choice"] == "required"
    assert result.citations[0].source_tier == "internal"


@pytest.mark.asyncio
async def test_external_question_routes_to_web_and_is_labelled_external() -> None:
    llm = ScriptedRouterLLM(tool_decision("web_search", query="EU AI Act status"))
    internal = StubInternal([_chunk()])
    web = FakeWebSearch([_web_result()])
    synth = ScriptedCompleter("According to an external source, the Act is in force [1].")
    router = _router(llm=llm, internal=internal, web=web, synth=synth)

    result = await router.answer("What is the current status of the EU AI Act?")

    assert result.tool_trace == ["web_search"]
    assert result.used_web_search is True
    assert internal.queries == []
    citation = result.citations[0]
    assert citation.source_tier == "external"
    assert citation.reference.url == "https://example.org/ai-act"
    assert citation.reference.retrieved_at == RETRIEVED_AT
    assert citation.reference.chunk_id is None
    # The synthesis context labels web sources as external.
    assert "Web (external): EU AI Act overview" in synth.prompts[0][1]


@pytest.mark.asyncio
async def test_greeting_routes_to_answer_directly_without_retrieval() -> None:
    llm = ScriptedRouterLLM(tool_decision("answer_directly"))
    internal = StubInternal([_chunk()])
    synth = ScriptedCompleter()
    router = _router(llm=llm, internal=internal, synth=synth, direct=ScriptedCompleter("Hi!"))

    result = await router.answer("hello")

    assert result.answer == "Hi!"
    assert result.tool_trace == ["answer_directly"]
    assert result.citations == []
    assert internal.queries == []
    assert synth.prompts == []


@pytest.mark.asyncio
async def test_misrouted_factual_direct_answer_falls_back_to_internal() -> None:
    llm = ScriptedRouterLLM(tool_decision("answer_directly"))
    internal = StubInternal([_chunk()])
    router = _router(llm=llm, internal=internal, direct=ScriptedCompleter(DIRECT_ANSWER_SENTINEL))

    result = await router.answer("Who approves expenses?")

    assert internal.queries == ["Who approves expenses?"]
    assert result.citations and result.citations[0].source_tier == "internal"


@pytest.mark.asyncio
async def test_router_llm_failure_falls_back_to_internal_search() -> None:
    internal = StubInternal([_chunk()])
    router = _router(llm=FailingRouterLLM(), internal=internal)

    result = await router.answer("Who approves expenses?")

    assert result.tool_trace == ["internal_search"]
    assert result.citations


@pytest.mark.asyncio
async def test_first_turn_without_tool_call_still_retrieves() -> None:
    llm = ScriptedRouterLLM(RouterDecision(content="Managers approve everything."))
    internal = StubInternal([_chunk()])
    router = _router(llm=llm, internal=internal)

    result = await router.answer("Who approves expenses?")

    assert result.tool_trace == ["internal_search"]
    assert "everything" not in result.answer


# --- guardrails + iteration cap -------------------------------------------------------


@pytest.mark.asyncio
async def test_irrelevant_internal_results_escalate_to_web() -> None:
    llm = ScriptedRouterLLM(tool_decision("internal_search"))
    internal = StubInternal([_chunk(rerank=0.05)])
    web = FakeWebSearch([_web_result()])
    router = _router(llm=llm, internal=internal, web=web, synth=ScriptedCompleter("In force [1]."))

    result = await router.answer("What is our policy on the EU AI Act?")

    assert result.tool_trace == ["internal_search", "web_search"]
    assert result.used_web_search is True
    # The below-threshold chunk is not offered as evidence; only the web result is.
    assert result.citations[0].source_tier == "external"


@pytest.mark.asyncio
async def test_rrf_fallback_without_rerank_scores_counts_as_relevant() -> None:
    llm = ScriptedRouterLLM(tool_decision("internal_search"))
    web = FakeWebSearch([_web_result()])
    router = _router(llm=llm, internal=StubInternal([_chunk(rerank=None)]), web=web)

    result = await router.answer("Who approves expenses?")

    assert result.tool_trace == ["internal_search"]
    assert web.queries == []


@pytest.mark.asyncio
async def test_iteration_cap_returns_low_confidence_without_guessing() -> None:
    # The model keeps asking for more searches; nothing relevant ever comes back.
    llm = ScriptedRouterLLM(*(tool_decision("internal_search") for _ in range(10)))
    internal = StubInternal([])
    web = FakeWebSearch([])
    synth = ScriptedCompleter("Fabricated answer [1].")
    router = _router(llm=llm, internal=internal, web=web, synth=synth, max_iterations=3)

    result = await router.answer("What is the airspeed of our CFO's parrot?")

    assert len(result.tool_trace) == 3
    assert result.answer == LOW_CONFIDENCE_ANSWER
    assert result.citations == []
    assert synth.prompts == []  # synthesis never ran, so nothing could be fabricated


@pytest.mark.asyncio
async def test_parallel_calls_are_trimmed_to_the_iteration_cap() -> None:
    llm = ScriptedRouterLLM(tool_decision("internal_search", "web_search", "internal_search"))
    router = _router(llm=llm, internal=StubInternal([]), web=FakeWebSearch([]), max_iterations=2)

    result = await router.answer("anything")

    assert result.tool_trace == ["internal_search", "web_search"]
    assert result.answer == LOW_CONFIDENCE_ANSWER


@pytest.mark.asyncio
async def test_web_failure_answers_from_internal_with_notice() -> None:
    llm = ScriptedRouterLLM(tool_decision("internal_search", "web_search"))
    router = _router(llm=llm, internal=StubInternal([_chunk()]), web=FakeWebSearch(failed=True))

    result = await router.answer("How do we approve expenses and what do regulators say?")

    assert result.used_web_search is False
    assert result.notices == [WEB_UNAVAILABLE_NOTICE]
    assert result.citations[0].source_tier == "internal"


class DownInternal(StubInternal):
    async def retrieve(self, question: str) -> tuple[list[RetrievedChunk], list[GraphFact]]:
        self.queries.append(question)
        raise RetrievalError("The query embedding service is unavailable.")


@pytest.mark.asyncio
async def test_failed_tools_are_not_retried_and_total_outage_raises_503() -> None:
    # Regression (live run 2026-09-30): the model re-requested web_search after it
    # had failed, and an internal outage was reported as "couldn't find an answer".
    llm = ScriptedRouterLLM(
        tool_decision("web_search"),
        tool_decision("internal_search"),
        tool_decision("web_search", "internal_search"),
    )
    internal = DownInternal()
    web = FakeWebSearch(failed=True)
    router = _router(llm=llm, internal=internal, web=web)

    with pytest.raises(RetrievalError):
        await router.answer("What is the latest Python release?")

    assert len(web.queries) == 1
    assert len(internal.queries) == 1


@pytest.mark.asyncio
async def test_web_failure_falls_back_to_internal_search() -> None:
    llm = ScriptedRouterLLM(*(tool_decision("web_search") for _ in range(3)))
    internal = StubInternal([_chunk()])
    web = FakeWebSearch(failed=True)
    router = _router(llm=llm, internal=internal, web=web)

    result = await router.answer("What does the latest regulation say about expenses?")

    assert result.tool_trace == ["web_search", "internal_search"]
    assert result.notices == [WEB_UNAVAILABLE_NOTICE]
    assert result.citations[0].source_tier == "internal"
    assert len(llm.calls) == 1  # no wasted router turns re-requesting a dead tool


@pytest.mark.asyncio
async def test_internal_outage_still_answers_from_web() -> None:
    llm = ScriptedRouterLLM(tool_decision("internal_search"))
    web = FakeWebSearch([_web_result()])
    router = _router(
        llm=llm, internal=DownInternal(), web=web, synth=ScriptedCompleter("In force [1].")
    )

    result = await router.answer("Is the EU AI Act in force?")

    assert result.tool_trace == ["internal_search", "web_search"]
    assert result.citations[0].source_tier == "external"


@pytest.mark.asyncio
async def test_graph_search_is_cancelled_when_vector_retrieval_fails() -> None:
    graph_started = asyncio.Event()
    graph_cancelled = asyncio.Event()

    class SlowGraph:
        async def search(self, question: str) -> list[GraphFact]:
            graph_started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                graph_cancelled.set()
                raise
            return []

    class FailingHybrid:
        async def search(self, question: str) -> list[RetrievedChunk]:
            await graph_started.wait()
            raise RetrievalError("embedding service down")

    service = HybridRagService(
        cast(Any, FailingHybrid()),
        FakeReranker(),
        AnswerSynthesizer(ScriptedCompleter()),
        cast(Any, SlowGraph()),
    )

    with pytest.raises(RetrievalError):
        await service.retrieve("q")
    await asyncio.sleep(0)
    assert graph_cancelled.is_set()


# --- provenance through the full pipeline ----------------------------------------------


@pytest.mark.asyncio
async def test_superseded_sop_citation_triggers_version_warning() -> None:
    llm = ScriptedRouterLLM(tool_decision("internal_search"))
    old = _chunk("Manager review above $250.", current=False, version="v2.1")
    router = _router(
        llm=llm, internal=StubInternal([old]), synth=ScriptedCompleter("Review above $250 [1].")
    )

    result = await router.answer("What was the v2.1 manager review threshold?")

    assert result.version_warning is not None
    assert "superseded" in result.version_warning.lower()
    assert result.citations[0].reference.is_current_version is False


@pytest.mark.asyncio
async def test_uncited_claim_is_dropped_through_the_router() -> None:
    llm = ScriptedRouterLLM(tool_decision("internal_search"))
    synth = ScriptedCompleter("Managers approve expenses [1]. The CFO signs everything.")
    router = _router(llm=llm, synth=synth)

    result = await router.answer("Who approves expenses?")

    assert "CFO" not in result.answer
    assert result.flagged_claims == ["The CFO signs everything."]


def test_web_results_are_never_persisted_to_vector_store(settings: Settings) -> None:
    class RecordingVectorStore(FakeVectorStore):
        def __init__(self) -> None:
            super().__init__()
            self.writes: list[str] = []

        async def upsert(self, *args: Any, **kwargs: Any) -> int:
            self.writes.append("upsert")
            return 0

        async def mark_superseded(self, *args: Any, **kwargs: Any) -> int:
            self.writes.append("mark_superseded")
            return 0

    class ConstantCompleter:
        # create_app shares one completion provider across synthesis, entity
        # extraction, rewrite, and direct answers, so the reply must not deplete.
        async def complete(self, *, system_prompt: str, user_prompt: str) -> CompletionResult:
            del system_prompt, user_prompt
            return CompletionResult(text="It is in force [2].")

    store = RecordingVectorStore()
    app = create_app(
        settings,
        vector_store=cast(Any, store),
        graph_store=cast(Any, FakeGraphStore()),
        embedding_provider=FakeEmbeddingProvider(),
        completion_provider=ConstantCompleter(),
        reranker=FakeReranker(),
        router_llm=ScriptedRouterLLM(tool_decision("internal_search", "web_search")),
        web_searcher=FakeWebSearch([_web_result()]),
    )
    with TestClient(app) as client:
        payload = client.post("/query", json={"question": "EU AI Act status?"}).json()

    assert payload["used_web_search"] is True
    assert payload["citations"][0]["source_tier"] == "external"
    assert payload["citations"][0]["reference"]["url"] == "https://example.org/ai-act"
    assert store.writes == []


# --- conversation memory -----------------------------------------------------------


@pytest.mark.asyncio
async def test_follow_up_is_rewritten_to_standalone_query_using_history() -> None:
    store = InMemoryConversationStore(max_turns=20, ttl=timedelta(hours=1))
    rewritten = "What was the manager review threshold in the v2.1 expense approval SOP?"
    rewriter = ScriptedCompleter(rewritten)
    llm = ScriptedRouterLLM(
        tool_decision("internal_search", query="manager review threshold"),
        # Turn 2: no query argument, so retrieval falls back to the rewrite.
        RouterDecision(tool_calls=[ToolCall(id="c0", name="internal_search", arguments={})]),
    )
    internal = StubInternal([_chunk()])
    router = _router(
        llm=llm,
        internal=internal,
        rewriter=rewriter,
        store=store,
        synth=ScriptedCompleter("Managers approve [1].", "Review above $250 [1]."),
    )

    first = await router.answer("What is the manager review threshold?")
    second = await router.answer("what about the previous version?", first.conversation_id)

    assert second.conversation_id == first.conversation_id
    assert internal.queries[-1] == rewritten
    # The router is given the standalone question, not the terse follow-up.
    assert llm.calls[-1]["messages"][1]["content"] == rewritten
    # The rewriter saw the prior turn; the first turn (no history) was not rewritten.
    assert len(rewriter.prompts) == 1
    assert "What is the manager review threshold?" in rewriter.prompts[0][1]


@pytest.mark.asyncio
async def test_conversation_history_is_isolated_by_id() -> None:
    store = InMemoryConversationStore(max_turns=20, ttl=timedelta(hours=1))
    synth = ScriptedCompleter(
        "Managers approve [1].", "Managers approve [1].", "Managers approve [1]."
    )
    router = _router(
        llm=ScriptedRouterLLM(*(tool_decision("internal_search") for _ in range(3))),
        synth=synth,
        store=store,
    )

    a = await router.answer("SECRET-TOPIC-ALPHA expenses?")
    b = await router.answer("Who approves expenses?", "not-a-real-id")
    b2 = await router.answer("And above $1000?", b.conversation_id)

    assert b.conversation_id not in {a.conversation_id, "not-a-real-id"}
    assert b2.conversation_id == b.conversation_id
    # Neither of conversation B's synthesis prompts contains conversation A's text.
    assert all("SECRET-TOPIC-ALPHA" not in prompt for _, prompt in synth.prompts[1:])
    assert len(await store.history(a.conversation_id, limit=10)) == 1
    assert len(await store.history(b.conversation_id, limit=10)) == 2


@pytest.mark.asyncio
async def test_prior_turn_text_is_not_a_citation_source() -> None:
    store = InMemoryConversationStore(max_turns=20, ttl=timedelta(hours=1))
    conv_id, _ = await store.resolve(None)
    await store.append(
        conv_id,
        ConversationTurn(
            question="Who approves refunds?",
            standalone_question="Who approves refunds?",
            answer="The CFO approves refunds.",
        ),
    )
    synth = ScriptedCompleter(
        # Claims resting only on history: one uncited, one pointing past the sources.
        "The CFO approves refunds. The CFO approves refunds [2]."
    )
    router = _router(
        llm=ScriptedRouterLLM(tool_decision("internal_search")),
        store=store,
        synth=synth,
        internal=StubInternal([_chunk()]),
    )

    result = await router.answer("Remind me who approves refunds?", conv_id)

    # History reached the synthesizer as background only (not a numbered source).
    assert "The CFO approves refunds." in synth.prompts[0][1]
    assert "[2]" not in synth.prompts[0][1]
    assert result.citations == []
    assert "CFO" not in result.answer
