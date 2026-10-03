"""Citation-grounded answer synthesis through Sarvam-105B."""

from __future__ import annotations

from typing import Any, Protocol

import httpx
import structlog
from pydantic import BaseModel
from sarvamai import AsyncSarvamAI
from sarvamai.core.api_error import ApiError
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from src.agent.prompts import SYNTHESIS_SYSTEM_PROMPT
from src.api.config import Settings
from src.api.errors import LLMError
from src.generation.provenance import CitableSource, GroundedAnswer, ground_answer
from src.indexing.graph_schema import GraphFact
from src.indexing.vector_store import RetrievedChunk
from src.retrieval.web_search import WebResult

logger = structlog.get_logger(__name__)

__all__ = [
    "SYNTHESIS_SYSTEM_PROMPT",
    "AnswerSynthesizer",
    "CompletionProvider",
    "CompletionResult",
    "SarvamCompletionProvider",
    "is_retryable_sarvam_error",
]


class _Unset:
    """Sentinel marking an unsupplied optional argument (distinct from ``None``)."""


_UNSET = _Unset()


class CompletionResult(BaseModel):
    """Vendor-neutral completion text and usage metadata."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class CompletionProvider(Protocol):
    async def complete(self, *, system_prompt: str, user_prompt: str) -> CompletionResult: ...


class SarvamCompletionProvider:
    """Small resilient boundary around the async Sarvam SDK."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: Any | None = None,
        reasoning_effort: str | _Unset | None = _UNSET,
        max_tokens: int | None = None,
    ) -> None:
        self._model = settings.llm_model
        # ``reasoning_effort`` defaults to the synthesis setting but can be
        # overridden (e.g. ``None`` to disable reasoning for extraction calls).
        # A sentinel distinguishes "not provided" from an explicit ``None``.
        self._reasoning_effort = (
            settings.llm_synth_reasoning_effort
            if isinstance(reasoning_effort, _Unset)
            else reasoning_effort
        )
        self._max_tokens = max_tokens or settings.llm_synth_max_tokens
        self._max_retries = settings.llm_max_retries
        self._client = client or AsyncSarvamAI(
            api_subscription_key=settings.sarvam_api_key.get_secret_value(),
            timeout=settings.llm_request_timeout_seconds,
        )

    async def complete(self, *, system_prompt: str, user_prompt: str) -> CompletionResult:
        """Call Sarvam, recovering once when reasoning exhausts the token budget.

        On ``sarvam-105b`` reasoning tokens are drawn from ``max_tokens`` before
        any visible content, so a tight budget can return HTTP 200 with
        ``finish_reason="length"`` and empty ``content`` (the reply is stranded
        in ``reasoning_content``). That is a budget problem, not an outage, so we
        retry once with a doubled ceiling before surfacing an error.
        """
        try:
            result = await self._complete_once(system_prompt, user_prompt, self._max_tokens)
            if result is not None:
                return result
            retry_budget = self._max_tokens * 2
            logger.warning(
                "sarvam_truncated_in_reasoning_retrying",
                max_tokens=self._max_tokens,
                retry_max_tokens=retry_budget,
            )
            result = await self._complete_once(system_prompt, user_prompt, retry_budget)
            if result is not None:
                return result
            raise LLMError(
                "Sarvam exhausted its token budget on reasoning before answering; "
                "raise LLM_SYNTH_MAX_TOKENS or disable synthesis reasoning."
            )
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError("Answer synthesis is temporarily unavailable.") from exc

    async def _complete_once(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> CompletionResult | None:
        """One retried Sarvam call; ``None`` means truncated-in-reasoning (empty content)."""
        response: Any | None = None
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=0.25, min=0.25, max=2),
            retry=retry_if_exception(is_retryable_sarvam_error),
            reraise=True,
        ):
            with attempt:
                response = await self._client.chat.completions(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    reasoning_effort=self._reasoning_effort,
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
        if response is None:
            raise LLMError("Sarvam did not produce a completion response.")
        choice = response.choices[0]
        content = choice.message.content
        if not content:
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason == "length":
                # Budget consumed by reasoning: recoverable, signal with None.
                return None
            # Empty for any other reason (e.g. content filter): a real failure.
            raise LLMError("Sarvam returned an empty answer.")
        usage = getattr(response, "usage", None)
        return CompletionResult(
            text=content,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )


def is_retryable_sarvam_error(exc: BaseException) -> bool:
    """Return whether a Sarvam failure is transient (network, 429, or 5xx)."""
    if isinstance(exc, (httpx.TransportError, TimeoutError)):
        return True
    return isinstance(exc, ApiError) and (
        exc.status_code == 429 or (exc.status_code is not None and exc.status_code >= 500)
    )


class AnswerSynthesizer:
    """Build numbered context, call the LLM, then enforce citation grounding."""

    def __init__(self, completion_provider: CompletionProvider) -> None:
        self._completion_provider = completion_provider

    async def synthesize(
        self,
        question: str,
        sources: list[RetrievedChunk],
        *,
        graph_facts: list[GraphFact] | None = None,
        web_results: list[WebResult] | None = None,
        conversation_context: str | None = None,
    ) -> GroundedAnswer:
        """Answer from numbered internal + external sources, then ground it.

        ``conversation_context`` is prior-turn text shown only to resolve the
        question; it is deliberately *not* numbered, so it can never be cited
        and any claim resting on it alone is dropped by the grounding step.
        """
        # Reranked chunks are numbered first (they carry the rerank ordering);
        # graph facts are appended after as additional citable sources. Graph
        # facts bypass the reranker deliberately - they are already precise,
        # structured triples, so spending rerank budget on them adds latency
        # without improving ordering.
        citable: list[CitableSource] = [CitableSource.from_chunk(source) for source in sources]
        # Map chunk_id -> retrieved chunk so a graph fact cites the real source
        # document (title/version) it was extracted from, when that chunk is in
        # the retrieved set, instead of a synthetic "Knowledge graph" label.
        chunk_lookup = {str(source.chunk.chunk_id): source for source in sources}
        citable.extend(
            CitableSource.from_graph_fact(fact, chunk_lookup=chunk_lookup)
            for fact in (graph_facts or [])
        )
        # External web results are numbered last so internal evidence keeps the
        # lowest (most prominent) markers.
        citable.extend(CitableSource.from_web(result) for result in (web_results or []))

        if not citable:
            return GroundedAnswer(
                answer="The provided context is insufficient to answer this question.",
                citations=[],
            )

        context = "\n\n".join(
            f"[{index}] {source.context_label()}\n{source.text}"
            for index, source in enumerate(citable, start=1)
        )
        parts: list[str] = []
        if conversation_context:
            parts.append(
                "Conversation so far (background only - NOT a source, never cite it):\n"
                f"{conversation_context}"
            )
        parts.append(f"Question:\n{question.strip()}")
        parts.append(f"Numbered sources:\n{context}")
        completion = await self._completion_provider.complete(
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            user_prompt="\n\n".join(parts),
        )
        grounded = ground_answer(completion.text, citable)
        logger.info(
            "answer_synthesized",
            source_count=len(sources),
            graph_fact_count=len(graph_facts or []),
            web_result_count=len(web_results or []),
            has_conversation_context=bool(conversation_context),
            citation_count=len(grounded.citations),
            flagged_claim_count=len(grounded.flagged_claims),
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )
        return grounded
