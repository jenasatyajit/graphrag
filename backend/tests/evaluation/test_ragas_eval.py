"""Tests for the RAGAS evaluator with a fully-faked judge and embeddings."""

from __future__ import annotations

from typing import ClassVar

import pytest

from src.api.config import Settings
from src.evaluation.golden_set import GoldenItem
from src.evaluation.ragas_eval import (
    MetricBundle,
    PipelineAnswer,
    RagasEvaluator,
    build_sarvam_judge,
)
from tests.conftest import settings_values


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


class _FakeAsyncOpenAI:
    """Capture AsyncOpenAI construction kwargs instead of opening a connection."""

    last_kwargs: ClassVar[dict[str, object]] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = kwargs


_FACTORY_CALLS: list[dict[str, object]] = []


def _fake_llm_factory(model: str, **kwargs: object) -> object:
    """Capture llm_factory args so the test can assert model params threaded through."""
    call = {"model": model, **kwargs}
    _FACTORY_CALLS.append(call)
    return call


def test_build_sarvam_judge_threads_max_tokens_and_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai
    import ragas.llms as ragas_llms

    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr(ragas_llms, "llm_factory", _fake_llm_factory)
    _FACTORY_CALLS.clear()

    settings = Settings(
        _env_file=None,
        **settings_values(),
        EVAL_JUDGE_MAX_TOKENS="2048",
        # The literal string "null" must map to Python None (reasoning disabled).
        EVAL_JUDGE_REASONING_EFFORT="null",
    )
    assert settings.eval_judge_reasoning_effort is None

    build_sarvam_judge(settings)

    # Model params are forwarded to llm_factory (ragas threads them into every
    # chat.completions.create call) with reasoning disabled.
    assert len(_FACTORY_CALLS) == 1
    factory_call = _FACTORY_CALLS[0]
    assert factory_call["model"] == settings.eval_judge_model
    assert factory_call["provider"] == "openai"
    assert isinstance(factory_call["client"], _FakeAsyncOpenAI)
    assert factory_call["max_tokens"] == 2048
    assert factory_call["reasoning_effort"] is None

    # The Sarvam auth caveat and base_url are preserved on the client.
    client_kwargs = _FakeAsyncOpenAI.last_kwargs
    key = settings.sarvam_api_key.get_secret_value()
    assert client_kwargs["base_url"] == str(settings.eval_judge_base_url)
    assert client_kwargs["api_key"] == key
    assert client_kwargs["default_headers"] == {"api-subscription-key": key}
    assert client_kwargs["timeout"] == settings.eval_judge_timeout_seconds
