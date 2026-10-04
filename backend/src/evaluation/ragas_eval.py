"""RAGAS golden-set metric evaluation with a Sarvam judge (Phase 6, SPEC 7.6).

Four RAGAS metrics are scored through the collections API (ragas 0.4.3):
faithfulness, context precision, context recall, and answer relevancy. The judge
LLM is Sarvam ``sarvam-105b`` reached through its OpenAI-compatible endpoint; the
answer-relevancy embeddings reuse the query-time BGE-M3 service via
:class:`HttpEmbeddingAdapter`.

The metric objects are injected (constructor / factory) so unit tests pass fakes
and never touch a vendor. A single metric call that fails - a judge timeout, a
transport error, a malformed response - is caught, logged, and recorded on the
item without the metric value; it degrades the score to ``None`` rather than
crashing the whole run (SPEC 13: graceful degradation).

Sarvam auth caveat: Sarvam authenticates with an ``api-subscription-key`` header,
not a bearer token, while the OpenAI SDK always sends ``Authorization: Bearer
<api_key>``. :func:`build_sarvam_judge` therefore passes the key BOTH ways - as
``api_key`` and as a ``default_headers`` subscription key - so whichever header
the ``/v1`` endpoint honors succeeds. The CLI runs one minimal judge call before
the full set to confirm auth before spending the run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field

from src.api.config import Settings
from src.evaluation.golden_set import GoldenItem
from src.evaluation.ragas_embeddings import HttpEmbeddingAdapter
from src.indexing.embeddings import EmbeddingProvider

if TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM

logger = structlog.get_logger(__name__)


@runtime_checkable
class ScoreResult(Protocol):
    """The shape RAGAS metric ``ascore`` returns: an object carrying ``.value``."""

    @property
    def value(self) -> float: ...


class FaithfulnessMetric(Protocol):
    async def ascore(
        self, *, user_input: str, response: str, retrieved_contexts: list[str]
    ) -> ScoreResult: ...


class ContextPrecisionMetric(Protocol):
    async def ascore(
        self, *, user_input: str, reference: str, retrieved_contexts: list[str]
    ) -> ScoreResult: ...


class ContextRecallMetric(Protocol):
    async def ascore(
        self, *, user_input: str, retrieved_contexts: list[str], reference: str
    ) -> ScoreResult: ...


class AnswerRelevancyMetric(Protocol):
    async def ascore(self, *, user_input: str, response: str) -> ScoreResult: ...


@dataclass
class MetricBundle:
    """The four injected RAGAS metric objects scored for every item.

    A plain dataclass (not a pydantic model): the fields hold arbitrary metric
    objects - real RAGAS collection metrics in production, fakes in tests - so
    they are kept as opaque, injectable dependencies rather than validated data.
    """

    faithfulness: FaithfulnessMetric
    context_precision: ContextPrecisionMetric
    context_recall: ContextRecallMetric
    answer_relevancy: AnswerRelevancyMetric


class MetricScores(BaseModel):
    """Per-item or aggregate metric values; ``None`` marks a failed/absent score."""

    faithfulness: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    answer_relevancy: float | None = None


class ItemScore(BaseModel):
    """One golden item's scores plus any per-metric error messages."""

    id: str
    category: str
    scores: MetricScores
    errors: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    """A pipeline's scored golden set: per-item results and the aggregate."""

    per_item: list[ItemScore]
    aggregate: MetricScores
    judge_model: str
    failures: list[str] = Field(default_factory=list)


class PipelineAnswer(BaseModel):
    """A pipeline's output for one golden item: the answer plus its contexts."""

    answer: str
    retrieved_contexts: list[str] = Field(default_factory=list)


def build_sarvam_judge(settings: Settings) -> InstructorBaseRagasLLM:
    """Construct the RAGAS judge LLM over Sarvam's OpenAI-compatible endpoint.

    Imports are local so importing this module (e.g. in a fakes-only unit test)
    does not require the ``openai``/``ragas`` LLM transport to be importable or a
    live key to be present. See the module docstring for the auth caveat.
    """
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    key = settings.sarvam_api_key.get_secret_value()
    client = AsyncOpenAI(
        base_url=str(settings.eval_judge_base_url),
        api_key=key,
        default_headers={"api-subscription-key": key},
        timeout=settings.eval_judge_timeout_seconds,
    )
    # ragas 0.4.3 forwards llm_factory kwargs into InstructorLLM.model_args and
    # straight into every ``chat.completions.create`` call. sarvam-105b spends
    # reasoning tokens from max_tokens BEFORE the verdict JSON, so a generous
    # budget plus reasoning disabled (reasoning_effort=None) keeps the structured
    # output from truncating. This mirrors the synthesizer's sarvam-105b handling.
    return llm_factory(
        settings.eval_judge_model,
        provider="openai",
        client=client,
        max_tokens=settings.eval_judge_max_tokens,
        reasoning_effort=settings.eval_judge_reasoning_effort,
    )


def build_default_metrics(
    judge_llm: InstructorBaseRagasLLM, embeddings: HttpEmbeddingAdapter
) -> MetricBundle:
    """Build the four real RAGAS collection metrics bound to the judge + embeddings."""
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    return MetricBundle(
        faithfulness=Faithfulness(llm=judge_llm),
        context_precision=ContextPrecision(llm=judge_llm),
        context_recall=ContextRecall(llm=judge_llm),
        answer_relevancy=AnswerRelevancy(llm=judge_llm, embeddings=embeddings),
    )


def build_embeddings_adapter(embedder: EmbeddingProvider) -> HttpEmbeddingAdapter:
    """Wrap the query-time embedding client as a RAGAS embeddings object."""
    return HttpEmbeddingAdapter(embedder)


class RagasEvaluator:
    """Score golden items against a pipeline's answers using injected metrics."""

    def __init__(
        self,
        metrics: MetricBundle,
        *,
        judge_model: str,
        max_concurrency: int = 4,
    ) -> None:
        self._metrics = metrics
        self._judge_model = judge_model
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))

    @property
    def judge_model(self) -> str:
        """The judge model string recorded in each result."""
        return self._judge_model

    async def score_item(self, item: GoldenItem, answer: PipelineAnswer) -> ItemScore:
        """Score one item across the four metrics, isolating per-metric failures."""
        contexts = answer.retrieved_contexts
        errors: list[str] = []

        faithfulness = await self._safe(
            "faithfulness",
            item.id,
            errors,
            self._metrics.faithfulness.ascore(
                user_input=item.question,
                response=answer.answer,
                retrieved_contexts=contexts,
            ),
        )
        context_precision = await self._safe(
            "context_precision",
            item.id,
            errors,
            self._metrics.context_precision.ascore(
                user_input=item.question,
                reference=item.ground_truth_answer,
                retrieved_contexts=contexts,
            ),
        )
        context_recall = await self._safe(
            "context_recall",
            item.id,
            errors,
            self._metrics.context_recall.ascore(
                user_input=item.question,
                retrieved_contexts=contexts,
                reference=item.ground_truth_answer,
            ),
        )
        answer_relevancy = await self._safe(
            "answer_relevancy",
            item.id,
            errors,
            self._metrics.answer_relevancy.ascore(
                user_input=item.question,
                response=answer.answer,
            ),
        )
        return ItemScore(
            id=item.id,
            category=item.category,
            scores=MetricScores(
                faithfulness=faithfulness,
                context_precision=context_precision,
                context_recall=context_recall,
                answer_relevancy=answer_relevancy,
            ),
            errors=errors,
        )

    async def evaluate(
        self,
        items: list[GoldenItem],
        answers: dict[str, PipelineAnswer],
    ) -> EvalResult:
        """Score every item (bounded concurrency) and aggregate non-None values.

        ``answers`` maps a golden item id to the pipeline output for it. Items
        without a corresponding answer are recorded as failures and skipped.
        """
        failures: list[str] = []
        scorable: list[GoldenItem] = []
        for item in items:
            if item.id in answers:
                scorable.append(item)
            else:
                failures.append(f"{item.id}: no pipeline answer produced")

        async def _run(item: GoldenItem) -> ItemScore:
            async with self._semaphore:
                return await self.score_item(item, answers[item.id])

        per_item = list(await asyncio.gather(*(_run(item) for item in scorable)))
        for score in per_item:
            failures.extend(f"{score.id}: {message}" for message in score.errors)
        return EvalResult(
            per_item=per_item,
            aggregate=_aggregate(per_item),
            judge_model=self._judge_model,
            failures=failures,
        )

    async def _safe(
        self,
        metric_name: str,
        item_id: str,
        errors: list[str],
        coro: Awaitable[ScoreResult],
    ) -> float | None:
        """Await a metric call, recording and swallowing a single failure."""
        try:
            result = await coro
            return float(result.value)
        except Exception as exc:  # a judge failure must not abort the run
            message = f"{metric_name} failed: {type(exc).__name__}: {exc}"
            errors.append(message)
            logger.warning(
                "ragas_metric_failed",
                metric=metric_name,
                item_id=item_id,
                error=str(exc),
            )
            return None


def _aggregate(per_item: list[ItemScore]) -> MetricScores:
    """Average each metric over the items that produced a value for it."""
    return MetricScores(
        faithfulness=_mean(score.scores.faithfulness for score in per_item),
        context_precision=_mean(score.scores.context_precision for score in per_item),
        context_recall=_mean(score.scores.context_recall for score in per_item),
        answer_relevancy=_mean(score.scores.answer_relevancy for score in per_item),
    )


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)
