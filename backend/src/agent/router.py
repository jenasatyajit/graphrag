"""Query orchestration: the Phase 5 tool-calling router plus ablation baselines.

``AgentRouter`` is the live ``POST /query`` path (SPEC 7.4). A Sarvam-105B
tool-calling loop (reasoning disabled) chooses ``internal_search``,
``web_search`` or ``answer_directly``; the SPEC routing rules are enforced in
code as guardrails rather than left to model judgment:

1. The first turn uses ``tool_choice="required"``. If the model fails, returns
   no usable tool call, or mis-uses ``answer_directly`` for a factual question,
   the router falls back to ``internal_search``.
2. After ``internal_search``, if no chunk clears ``INTERNAL_RELEVANCE_THRESHOLD``
   (reranker relevance score) and the web has not been tried, the router forces
   a ``web_search`` itself. A chunk with no rerank score (reranker fell back to
   RRF order) counts as relevant, because an outage says nothing about relevance
   and should not silently spend Exa credits.
3. Total tool calls are capped at ``MAX_AGENT_ITERATIONS``. With no usable
   evidence after the loop, a fixed "couldn't find a confident answer" reply is
   returned and synthesis is skipped, so nothing can be guessed.
4. Every tool call is logged with a result summary.

The router only *selects* evidence; one synthesis call writes the final answer,
and provenance grounding drops any uncited claim. ``NaiveRagService`` and
``HybridRagService`` remain as Phase 6 ablation baselines.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.agent.llm import RouterDecision, ToolCall, ToolCallingProvider
from src.agent.memory import (
    ConversationStore,
    ConversationTurn,
    FollowUpRewriter,
    render_history,
)
from src.agent.prompts import (
    DIRECT_ANSWER_SENTINEL,
    DIRECT_ANSWER_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
)
from src.agent.tools import (
    ANSWER_DIRECTLY,
    INTERNAL_SEARCH,
    ROUTER_TOOLS,
    TOOL_NAMES,
    WEB_SEARCH,
)
from src.api.errors import AppError, LLMError, NotFoundError, RetrievalError
from src.api.schemas import Citation, IngestionSummary
from src.generation.provenance import GroundedAnswer
from src.generation.synthesizer import AnswerSynthesizer, CompletionProvider
from src.indexing.embeddings import EmbeddingProvider
from src.indexing.graph_schema import GraphFact
from src.indexing.vector_store import RetrievedChunk, VectorStore
from src.retrieval.graph_search import GraphSearchService
from src.retrieval.hybrid_search import HybridSearchService
from src.retrieval.reranker import Reranker
from src.retrieval.web_search import WebResult, WebSearcher

logger = structlog.get_logger(__name__)

LOW_CONFIDENCE_ANSWER = (
    "I couldn't find a confident answer to that in the company documents or from a live web search."
)
WEB_UNAVAILABLE_NOTICE = "Live web search was unavailable for this request."
_EXCERPT_CHARS = 200


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

    async def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Embed the question and run dense-only retrieval.

        Exposes the Phase 6 ablation hook: this is the exact retrieval ``query``
        performs, factored out so the ablation runner can capture the dense
        contexts without a second round-trip. ``query`` calls it, so its output
        is unchanged (behavior-preserving refactor).
        """
        if self._embedder is None:
            raise RetrievalError("The query embedding service is not configured.")
        embedding = await self._embedder.embed([question], mode="query")
        candidates = await self._vector_store.dense_search(embedding.dense[0])
        logger.info(
            "dense_retrieval_completed",
            candidate_count=len(candidates),
            chunk_ids=[str(candidate.chunk.chunk_id) for candidate in candidates],
        )
        return candidates

    async def query(self, question: str) -> GroundedAnswer:
        candidates = await self.retrieve(question)
        return await self._synthesizer.synthesize(question, candidates)


class HybridRagService:
    """Internal retrieval: hybrid search + rerank + graph facts.

    ``retrieve`` is the body of the router's ``internal_search`` tool; ``query``
    adds synthesis and remains the Phase 2-4 pipeline for the ablation study.
    Hybrid and graph retrieval run concurrently; graph search never raises (it
    degrades to no facts), so a graph outage lowers quality without failing.
    """

    def __init__(
        self,
        hybrid_search: HybridSearchService,
        reranker: Reranker,
        synthesizer: AnswerSynthesizer,
        graph_search: GraphSearchService | None = None,
    ) -> None:
        self._hybrid_search = hybrid_search
        self._reranker = reranker
        self._synthesizer = synthesizer
        self._graph_search = graph_search

    async def retrieve(self, question: str) -> tuple[list[RetrievedChunk], list[GraphFact]]:
        async def _graph() -> list[GraphFact]:
            return await self._graph_search.search(question) if self._graph_search else []

        async def _vector() -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
            candidates = await self._hybrid_search.search(question)
            return candidates, await self._reranker.rerank(question, candidates)

        graph_task = asyncio.create_task(_graph())
        try:
            candidates, reranked = await _vector()
        except BaseException:
            # Don't leave graph work (and its LLM call) running after a failure.
            graph_task.cancel()
            raise
        graph_facts = await graph_task
        logger.info(
            "hybrid_rerank_completed",
            candidate_count=len(candidates),
            reranked_count=len(reranked),
            graph_fact_count=len(graph_facts),
            chunk_ids=[str(candidate.chunk.chunk_id) for candidate in reranked],
        )
        return reranked, graph_facts

    async def query(self, question: str) -> GroundedAnswer:
        reranked, graph_facts = await self.retrieve(question)
        return await self._synthesizer.synthesize(question, reranked, graph_facts=graph_facts)


class AgentAnswer(BaseModel):
    """Everything ``POST /query`` returns (SPEC 8 plus conversation fields)."""

    answer: str
    citations: list[Citation]
    used_web_search: bool
    version_warning: str | None
    tool_trace: list[str]
    conversation_id: str
    notices: list[str] = Field(default_factory=list)
    flagged_claims: list[str] = Field(default_factory=list)


class _EvidencePool:
    """Per-request evidence accumulated across tool calls (never persisted)."""

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold
        self.chunks: dict[str, RetrievedChunk] = {}
        self.facts: dict[tuple[str, str, str], GraphFact] = {}
        self.web: dict[str, WebResult] = {}
        self.internal_relevant = False
        self.internal_error: AppError | None = None
        self.web_attempted = False
        self.web_failed = False
        self.direct_text: str | None = None

    def is_relevant(self, chunk: RetrievedChunk) -> bool:
        return chunk.rerank_score is None or chunk.rerank_score >= self._threshold

    def add_internal(self, chunks: list[RetrievedChunk], facts: list[GraphFact]) -> list[bool]:
        flags = [self.is_relevant(chunk) for chunk in chunks]
        for chunk, relevant in zip(chunks, flags, strict=True):
            if not relevant:
                continue
            key = str(chunk.chunk.chunk_id)
            existing = self.chunks.get(key)
            if existing is None or (chunk.rerank_score or 0.0) > (existing.rerank_score or 0.0):
                self.chunks[key] = chunk
        for fact in facts:
            self.facts.setdefault((fact.subject_name, fact.relation, fact.object_name), fact)
        self.internal_relevant = self.internal_relevant or any(flags)
        return flags

    def add_web(self, results: list[WebResult]) -> None:
        for result in results:
            self.web.setdefault(result.url, result)

    @property
    def sufficient(self) -> bool:
        return self.internal_relevant or bool(self.web)

    @property
    def has_evidence(self) -> bool:
        return bool(self.chunks or self.facts or self.web)

    def ranked_chunks(self, limit: int) -> list[RetrievedChunk]:
        ordered = sorted(
            self.chunks.values(),
            key=lambda item: item.rerank_score if item.rerank_score is not None else -1.0,
            reverse=True,
        )
        return ordered[:limit]


class AgentRouter:
    """Tool-calling router with guardrails, conversation memory, and synthesis."""

    def __init__(
        self,
        *,
        router_llm: ToolCallingProvider,
        internal: HybridRagService,
        web: WebSearcher,
        synthesizer: AnswerSynthesizer,
        direct_responder: CompletionProvider,
        conversation_store: ConversationStore,
        rewriter: FollowUpRewriter,
        max_iterations: int,
        relevance_threshold: float,
        context_turns: int,
        max_internal_sources: int,
    ) -> None:
        self._llm = router_llm
        self._internal = internal
        self._web = web
        self._synthesizer = synthesizer
        self._direct = direct_responder
        self._store = conversation_store
        self._rewriter = rewriter
        self._max_iterations = max_iterations
        self._threshold = relevance_threshold
        self._context_turns = context_turns
        self._max_internal_sources = max_internal_sources

    async def answer(self, question: str, conversation_id: str | None = None) -> AgentAnswer:
        conv_id, is_new = await self._store.resolve(conversation_id)
        history = [] if is_new else await self._store.history(conv_id, limit=self._context_turns)
        standalone = await self._rewriter.rewrite(question, history)
        logger.info(
            "query_received",
            conversation_id=conv_id,
            new_conversation=is_new,
            history_turns=len(history),
            rewritten=standalone != question,
        )

        pool = _EvidencePool(self._threshold)
        trace: list[str] = []
        await self._route(standalone, pool, trace)
        result = await self._finish(standalone, pool, trace, conv_id, history)

        try:
            await self._store.append(
                conv_id,
                ConversationTurn(
                    question=question,
                    standalone_question=standalone,
                    answer=result.answer,
                    tool_trace=result.tool_trace,
                    used_web_search=result.used_web_search,
                ),
            )
        except Exception:
            # Memory is a convenience; a store failure must not lose the answer.
            logger.warning("conversation_append_failed", conversation_id=conv_id, exc_info=True)
        return result

    # --- routing loop ---------------------------------------------------------

    async def _route(self, question: str, pool: _EvidencePool, trace: list[str]) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        turn = 0
        # Router turns are bounded as well as tool calls, because a request for a
        # tool that already failed this request is answered without executing it.
        while len(trace) < self._max_iterations and turn < self._max_iterations:
            decision = await self._decide(messages, first=turn == 0)
            turn += 1
            calls = [] if decision is None else _usable_calls(decision.tool_calls)
            if not calls:
                if not trace:
                    # Guardrail: never skip retrieval on the first turn.
                    await self._forced(INTERNAL_SEARCH, question, question, pool, trace, messages)
                    await self._escalate_to_web(question, pool, trace, messages)
                break

            calls = calls[: self._max_iterations - len(trace)]
            messages.append(RouterDecision(tool_calls=calls).assistant_message())
            runnable = [call for call in calls if not _already_failed(call.name, pool)]
            summaries = dict(
                zip(
                    (call.id for call in runnable),
                    await asyncio.gather(*(self._execute(c, question, pool) for c in runnable)),
                    strict=True,
                )
            )
            for call in calls:
                if call.id in summaries:
                    trace.append(call.name)
                    summary = summaries[call.id]
                else:
                    logger.info("router_skipped_failed_tool", tool=call.name)
                    summary = {"tool": call.name, "error": "unavailable for this request"}
                messages.append(_tool_message(call.id, summary))

            if pool.direct_text is not None and not any(c.name != ANSWER_DIRECTLY for c in calls):
                break
            await self._escalate_to_web(question, pool, trace, messages)
            await self._fall_back_to_internal(question, pool, trace, messages)
            if pool.sufficient or not runnable:
                # Done, or the model only asked for tools that already failed.
                break

        if len(trace) >= self._max_iterations and not pool.sufficient:
            logger.info("router_iteration_cap_reached", tool_calls=len(trace))

    async def _decide(
        self, messages: list[dict[str, Any]], *, first: bool
    ) -> RouterDecision | None:
        try:
            decision = await self._llm.decide(
                messages,
                tools=ROUTER_TOOLS,
                tool_choice="required" if first else "auto",
            )
        except LLMError:
            logger.warning("router_llm_failed", exc_info=True)
            return None
        logger.info(
            "router_decision",
            tool_calls=[call.name for call in decision.tool_calls],
            has_content=bool(decision.content),
            input_tokens=decision.input_tokens,
            output_tokens=decision.output_tokens,
        )
        return decision

    async def _escalate_to_web(
        self,
        question: str,
        pool: _EvidencePool,
        trace: list[str],
        messages: list[dict[str, Any]],
    ) -> None:
        """Guardrail 2: force web search when internal search found nothing relevant."""
        internal_tried = INTERNAL_SEARCH in trace
        if (
            internal_tried
            and not pool.internal_relevant
            and not pool.web_attempted
            and len(trace) < self._max_iterations
        ):
            logger.info("router_guardrail_web_escalation")
            await self._forced(WEB_SEARCH, question, question, pool, trace, messages)

    async def _fall_back_to_internal(
        self,
        question: str,
        pool: _EvidencePool,
        trace: list[str],
        messages: list[dict[str, Any]],
    ) -> None:
        """Guardrail: if the web failed and internal was never tried, try internal."""
        if (
            pool.web_failed
            and INTERNAL_SEARCH not in trace
            and not pool.sufficient
            and len(trace) < self._max_iterations
        ):
            logger.info("router_guardrail_internal_fallback")
            await self._forced(INTERNAL_SEARCH, question, question, pool, trace, messages)

    async def _forced(
        self,
        name: str,
        query: str,
        question: str,
        pool: _EvidencePool,
        trace: list[str],
        messages: list[dict[str, Any]],
    ) -> None:
        """Run a guardrail-chosen tool and record it as if the model had called it."""
        call = ToolCall(id=f"guardrail_{len(trace)}", name=name, arguments={"query": query})
        messages.append(RouterDecision(tool_calls=[call]).assistant_message())
        summary = await self._execute(call, question, pool, forced=True)
        trace.append(name)
        messages.append(_tool_message(call.id, summary))

    # --- tool execution -------------------------------------------------------

    async def _execute(
        self,
        call: ToolCall,
        question: str,
        pool: _EvidencePool,
        *,
        forced: bool = False,
    ) -> dict[str, Any]:
        query = call.arguments.get("query")
        query = query.strip() if isinstance(query, str) and query.strip() else question
        if call.name == INTERNAL_SEARCH:
            summary = await self._run_internal(query, pool)
        elif call.name == WEB_SEARCH:
            summary = await self._run_web(query, pool)
        else:
            summary = await self._run_direct(question, pool)
        logger.info(
            "router_tool_call",
            tool=call.name,
            forced=forced,
            query_length=len(query),
            result=summary,
        )
        return summary

    async def _run_internal(self, query: str, pool: _EvidencePool) -> dict[str, Any]:
        try:
            chunks, facts = await self._internal.retrieve(query)
        except AppError as exc:
            pool.internal_error = exc
            return {"tool": INTERNAL_SEARCH, "error": "internal search unavailable"}
        flags = pool.add_internal(chunks, facts)
        scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
        return {
            "tool": INTERNAL_SEARCH,
            "relevant": any(flags),
            "result_count": len(chunks),
            "top_rerank_score": round(max(scores), 4) if scores else None,
            "results": [
                {
                    "title": c.chunk.doc_title,
                    "sop_version": c.chunk.sop_version,
                    "is_current_version": c.chunk.is_current_version,
                    "relevant": relevant,
                    "excerpt": c.chunk.text[:_EXCERPT_CHARS],
                }
                for c, relevant in zip(chunks, flags, strict=True)
            ],
            "graph_facts": [fact.as_text() for fact in facts[:5]],
        }

    async def _run_web(self, query: str, pool: _EvidencePool) -> dict[str, Any]:
        pool.web_attempted = True
        outcome = await self._web.search(query)
        if outcome.failed:
            pool.web_failed = True
            return {"tool": WEB_SEARCH, "error": "web search unavailable"}
        pool.add_web(outcome.results)
        return {
            "tool": WEB_SEARCH,
            "result_count": len(outcome.results),
            "results": [
                {"title": r.title, "url": r.url, "excerpt": r.snippet[:_EXCERPT_CHARS]}
                for r in outcome.results
            ],
        }

    async def _run_direct(self, question: str, pool: _EvidencePool) -> dict[str, Any]:
        try:
            completion = await self._direct.complete(
                system_prompt=DIRECT_ANSWER_SYSTEM_PROMPT,
                user_prompt=question,
            )
            text = completion.text.strip()
        except Exception:
            logger.warning("direct_answer_failed", exc_info=True)
            text = ""
        if not text or DIRECT_ANSWER_SENTINEL in text:
            # Guardrail: a factual question mis-routed to answer_directly gets retrieval.
            logger.info("router_guardrail_direct_to_internal")
            summary = await self._run_internal(question, pool)
            return {"tool": ANSWER_DIRECTLY, "redirected_to": INTERNAL_SEARCH, **summary}
        pool.direct_text = text
        return {"tool": ANSWER_DIRECTLY, "answered": True}

    # --- final answer ---------------------------------------------------------

    async def _finish(
        self,
        question: str,
        pool: _EvidencePool,
        trace: list[str],
        conv_id: str,
        history: list[ConversationTurn],
    ) -> AgentAnswer:
        notices = [WEB_UNAVAILABLE_NOTICE] if pool.web_failed else []

        if pool.direct_text is not None and not pool.has_evidence:
            return AgentAnswer(
                answer=pool.direct_text,
                citations=[],
                used_web_search=False,
                version_warning=None,
                tool_trace=trace,
                conversation_id=conv_id,
                notices=notices,
            )

        if not pool.has_evidence:
            if pool.internal_error is not None and (not pool.web_attempted or pool.web_failed):
                # Internal retrieval is down and the web gave nothing to fall back
                # on: surface the dependency failure (503) instead of claiming the
                # answer simply wasn't found.
                raise pool.internal_error
            logger.info("router_low_confidence", tool_calls=len(trace))
            return AgentAnswer(
                answer=LOW_CONFIDENCE_ANSWER,
                citations=[],
                used_web_search=False,
                version_warning=None,
                tool_trace=trace,
                conversation_id=conv_id,
                notices=notices,
            )

        web_results = list(pool.web.values())
        grounded = await self._synthesizer.synthesize(
            question,
            pool.ranked_chunks(self._max_internal_sources),
            graph_facts=list(pool.facts.values()),
            web_results=web_results,
            conversation_context=render_history(history) or None,
        )
        return AgentAnswer(
            answer=grounded.answer,
            citations=grounded.citations,
            used_web_search=bool(web_results),
            version_warning=grounded.version_warning,
            tool_trace=trace,
            conversation_id=conv_id,
            notices=notices,
            flagged_claims=grounded.flagged_claims,
        )


def _already_failed(name: str, pool: _EvidencePool) -> bool:
    """A dependency that failed once this request is not called again."""
    if name == WEB_SEARCH:
        return pool.web_failed
    if name == INTERNAL_SEARCH:
        return pool.internal_error is not None
    return False


def _usable_calls(calls: list[ToolCall]) -> list[ToolCall]:
    usable = [call for call in calls if call.name in TOOL_NAMES]
    # answer_directly mixed with a search call is meaningless; keep the search.
    if any(call.name != ANSWER_DIRECTLY for call in usable):
        usable = [call for call in usable if call.name != ANSWER_DIRECTLY]
    return usable


def _tool_message(call_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(summary, default=str)}


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
