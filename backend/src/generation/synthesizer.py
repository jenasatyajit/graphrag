"""Citation-grounded answer synthesis through Sarvam-105B."""

from __future__ import annotations

from typing import Any, Protocol

import httpx
import structlog
from pydantic import BaseModel
from sarvamai import AsyncSarvamAI
from sarvamai.core.api_error import ApiError
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from src.api.config import Settings
from src.api.errors import LLMError
from src.generation.provenance import CitableSource, GroundedAnswer, ground_internal_answer
from src.indexing.graph_schema import GraphFact
from src.indexing.vector_store import RetrievedChunk

logger = structlog.get_logger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """Answer the user's question using ONLY the provided context.

Citation rules (follow exactly):
- End every sentence that states a fact with one or more citation markers like [1] or [1][3].
- Put the marker at the END of the sentence, after the period is not required - place it
  right before the period, e.g. "Managers approve expenses [1]."
- Do NOT write phrases like "document [1]" or "source [2]" inside the sentence; the marker
  alone is the reference.
- Write in plain prose sentences, not bullet lists. Do not number your sentences.
- If a claim cannot be supported by the context, omit it. If the context is insufficient to
  answer, reply exactly: "Insufficient context to answer." Never use your own knowledge."""


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
        response: Any | None = None
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=0.25, min=0.25, max=2),
                retry=retry_if_exception(_is_retryable_sarvam_error),
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
                        max_tokens=self._max_tokens,
                        temperature=0.1,
                    )
            if response is None:
                raise LLMError("Sarvam did not produce a completion response.")
            content = response.choices[0].message.content
            if not content:
                raise LLMError("Sarvam returned an empty answer.")
            usage = getattr(response, "usage", None)
            return CompletionResult(
                text=content,
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
            )
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError("Answer synthesis is temporarily unavailable.") from exc


def _is_retryable_sarvam_error(exc: BaseException) -> bool:
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
    ) -> GroundedAnswer:
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

        if not citable:
            return GroundedAnswer(
                answer="The provided context is insufficient to answer this question.",
                citations=[],
            )

        context = "\n\n".join(
            f"[{index}] Document: {source.doc_title}\n{source.text}"
            for index, source in enumerate(citable, start=1)
        )
        completion = await self._completion_provider.complete(
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=f"Question:\n{question.strip()}\n\nNumbered sources:\n{context}",
        )
        grounded = ground_internal_answer(completion.text, citable)
        logger.info(
            "answer_synthesized",
            source_count=len(sources),
            graph_fact_count=len(graph_facts or []),
            citation_count=len(grounded.citations),
            flagged_claim_count=len(grounded.flagged_claims),
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )
        return grounded
