"""Tests for the four-stage ablation runner using fakes for every component."""

from __future__ import annotations

from typing import Any, cast

import pytest

from src.api.config import Settings
from src.evaluation.ablation import (
    ALL_STAGES,
    PipelineBundle,
    run_ablation,
)
from src.evaluation.golden_set import GoldenItem, GoldenSet
from src.evaluation.ragas_eval import MetricBundle, RagasEvaluator
from src.generation.synthesizer import AnswerSynthesizer
from src.indexing.graph_schema import GraphFact
from tests.conftest import (
    FakeCompletionProvider,
    FakeEmbeddingProvider,
    FakeReranker,
    FakeVectorStore,
    settings_values,
)
from tests.evaluation.test_ragas_eval import FakeMetric


class FakeGraphSearch:
    """Return one deterministic fact so graph stages have graph contexts."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def search(self, question: str) -> list[GraphFact]:
        self.calls.append(question)
        return [
            GraphFact(
                subject_name="Finance Approval",
                relation="REQUIRES",
                object_name="SAP",
                chunk_ids=["00000000-0000-0000-0000-000000000001"],
            )
        ]


def _bundle(graph_search: FakeGraphSearch) -> PipelineBundle:
    synthesizer = AnswerSynthesizer(FakeCompletionProvider())
    return PipelineBundle(
        embedder=cast(Any, FakeEmbeddingProvider()),
        vector_store=cast(Any, FakeVectorStore()),
        reranker=cast(Any, FakeReranker()),
        synthesizer=synthesizer,
        graph_search=cast(Any, graph_search),
    )


def _metrics() -> MetricBundle:
    return MetricBundle(
        faithfulness=cast(Any, FakeMetric(0.9)),
        context_precision=cast(Any, FakeMetric(0.8)),
        context_recall=cast(Any, FakeMetric(0.7)),
        answer_relevancy=cast(Any, FakeMetric(0.6)),
    )


def _golden() -> GoldenSet:
    return GoldenSet(
        items=[
            GoldenItem(
                id=f"item-{i}",
                question=f"Question {i}?",
                ground_truth_answer=f"Reference {i}.",
                ground_truth_contexts=[f"Context {i}."],
                category="general",
            )
            for i in range(3)
        ]
    )


def _settings() -> Settings:
    return Settings(_env_file=None, **settings_values())


@pytest.mark.asyncio
async def test_all_four_stages_produce_a_complete_table() -> None:
    graph_search = FakeGraphSearch()
    evaluator = RagasEvaluator(_metrics(), judge_model="sarvam-105b")

    report = await run_ablation(_settings(), _golden(), evaluator, _bundle(graph_search))

    assert [stage.stage for stage in report.stages] == list(ALL_STAGES)
    assert report.item_count == 3
    assert report.judge_model == "sarvam-105b"
    for stage in report.stages:
        assert stage.result.aggregate.faithfulness is not None
        assert stage.result.aggregate.context_precision is not None
        assert stage.result.aggregate.context_recall is not None
        assert stage.result.aggregate.answer_relevancy is not None
        assert len(stage.result.per_item) == 3


@pytest.mark.asyncio
async def test_graph_stages_invoke_graph_search_and_naive_does_not() -> None:
    graph_search = FakeGraphSearch()
    evaluator = RagasEvaluator(_metrics(), judge_model="sarvam-105b")

    await run_ablation(_settings(), _golden(), evaluator, _bundle(graph_search))

    # naive + hybrid_rerank (graph disabled) make no graph calls; graph +
    # graph_multimodal each query graph search once per item (3 items each).
    assert len(graph_search.calls) == 6


@pytest.mark.asyncio
async def test_stage_subset_can_be_selected() -> None:
    graph_search = FakeGraphSearch()
    evaluator = RagasEvaluator(_metrics(), judge_model="sarvam-105b")

    report = await run_ablation(
        _settings(),
        _golden(),
        evaluator,
        _bundle(graph_search),
        stages=("naive", "graph"),
    )

    assert [stage.stage for stage in report.stages] == ["naive", "graph"]
    # The first stage has no previous stage, so its deltas are empty.
    assert report.stages[0].delta.faithfulness is None
    # The second stage records a delta vs the first.
    assert report.stages[1].delta.faithfulness == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_corpus_notes_reflect_multimodal_flag() -> None:
    graph_search = FakeGraphSearch()
    evaluator = RagasEvaluator(_metrics(), judge_model="sarvam-105b")
    settings = Settings(_env_file=None, **settings_values(), EVAL_CORPUS_INCLUDES_MULTIMODAL="true")

    report = await run_ablation(settings, _golden(), evaluator, _bundle(graph_search))

    assert "INCLUDES" in report.corpus_notes
