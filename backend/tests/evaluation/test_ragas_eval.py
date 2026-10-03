"""Tests for the RAGAS evaluator with a fully-faked judge and embeddings."""

from __future__ import annotations

import pytest

from src.evaluation.golden_set import GoldenItem
from src.evaluation.ragas_eval import (
    MetricBundle,
    PipelineAnswer,
    RagasEvaluator,
)


class _Score:
    """Stand-in for ragas MetricResult: exposes a ``.value``."""

    def __init__(self, value: float) -> None:
        self.value = value


class FakeMetric:
    """Deterministic metric returning a fixed score (or raising for a given item)."""

    def __init__(self, value: float, *, raise_on: str | None = None) -> None:
        self._value = value
        self._raise_on = raise_on
        self.calls: list[dict[str, object]] = []

    async def ascore(self, **kwargs: object) -> _Score:
        self.calls.append(kwargs)
        if self._raise_on is not None and kwargs.get("user_input") == self._raise_on:
            raise RuntimeError("judge timeout")
        return _Score(self._value)


def _bundle(**overrides: FakeMetric) -> MetricBundle:
    defaults = {
        "faithfulness": FakeMetric(0.9),
        "context_precision": FakeMetric(0.8),
        "context_recall": FakeMetric(0.7),
        "answer_relevancy": FakeMetric(0.6),
    }
    defaults.update(overrides)
    return MetricBundle(**defaults)  # type: ignore[arg-type]


def _items() -> list[GoldenItem]:
    return [
        GoldenItem(
            id=f"item-{i}",
            question=f"Question {i}?",
            ground_truth_answer=f"Reference {i}.",
            ground_truth_contexts=[f"Context {i}."],
            category="general",
        )
        for i in range(3)
    ]


def _answers(items: list[GoldenItem]) -> dict[str, PipelineAnswer]:
    return {
        item.id: PipelineAnswer(answer=f"Answer {item.id}.", retrieved_contexts=[f"ctx {item.id}"])
        for item in items
    }


@pytest.mark.asyncio
async def test_evaluate_returns_per_item_and_aggregate() -> None:
    items = _items()
    evaluator = RagasEvaluator(_bundle(), judge_model="sarvam-105b")

    result = await evaluator.evaluate(items, _answers(items))

    assert result.judge_model == "sarvam-105b"
    assert len(result.per_item) == 3
    assert result.aggregate.faithfulness == pytest.approx(0.9)
    assert result.aggregate.context_precision == pytest.approx(0.8)
    assert result.aggregate.context_recall == pytest.approx(0.7)
    assert result.aggregate.answer_relevancy == pytest.approx(0.6)
    assert result.failures == []


@pytest.mark.asyncio
async def test_score_item_passes_reference_to_context_metrics() -> None:
    items = _items()
    precision = FakeMetric(0.8)
    recall = FakeMetric(0.7)
    evaluator = RagasEvaluator(
        _bundle(context_precision=precision, context_recall=recall),
        judge_model="sarvam-105b",
    )

    await evaluator.evaluate(items, _answers(items))

    assert all(call["reference"] == f"Reference {i}." for i, call in enumerate(precision.calls))
    assert all(call["reference"] == f"Reference {i}." for i, call in enumerate(recall.calls))


@pytest.mark.asyncio
async def test_single_metric_failure_is_recorded_and_does_not_abort() -> None:
    items = _items()
    failing = FakeMetric(0.9, raise_on="Question 1?")
    evaluator = RagasEvaluator(_bundle(faithfulness=failing), judge_model="sarvam-105b")

    result = await evaluator.evaluate(items, _answers(items))

    # The run completes for every item.
    assert len(result.per_item) == 3
    failed = next(score for score in result.per_item if score.id == "item-1")
    assert failed.scores.faithfulness is None
    assert any("faithfulness failed" in message for message in failed.errors)
    # Other metrics for the same item still scored.
    assert failed.scores.context_precision == pytest.approx(0.8)
    # The failure is surfaced at the run level.
    assert any("item-1" in failure for failure in result.failures)
    # The aggregate averages only the two successful faithfulness scores.
    assert result.aggregate.faithfulness == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_missing_answer_is_recorded_as_failure() -> None:
    items = _items()
    answers = _answers(items)
    del answers["item-2"]
    evaluator = RagasEvaluator(_bundle(), judge_model="sarvam-105b")

    result = await evaluator.evaluate(items, answers)

    assert len(result.per_item) == 2
    assert any(
        "item-2" in failure and "no pipeline answer" in failure for failure in result.failures
    )


@pytest.mark.asyncio
async def test_aggregate_is_none_when_metric_never_succeeds() -> None:
    items = _items()

    class AlwaysFails(FakeMetric):
        async def ascore(self, **kwargs: object) -> _Score:
            raise RuntimeError("down")

    evaluator = RagasEvaluator(
        _bundle(answer_relevancy=AlwaysFails(0.0)),
        judge_model="sarvam-105b",
    )

    result = await evaluator.evaluate(items, _answers(items))

    assert result.aggregate.answer_relevancy is None
