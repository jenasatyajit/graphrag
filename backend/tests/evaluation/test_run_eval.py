"""End-to-end harness proof: ablation -> report writing with mocked components.

The real ``scripts/run_eval.py`` needs live cloud stores + the Sarvam judge, so
CI cannot run it. This test exercises the exact core the script orchestrates
(``run_ablation`` -> ``write_report``) with fully mocked components, proving the
4-stage table is produced and both report files are written and parseable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from src.api.config import Settings
from src.evaluation.ablation import ALL_STAGES, PipelineBundle, run_ablation
from src.evaluation.golden_set import GoldenSet, load_golden_set
from src.evaluation.ragas_eval import MetricBundle, RagasEvaluator
from src.evaluation.report import write_report
from src.generation.synthesizer import AnswerSynthesizer, CompletionResult
from src.indexing.graph_schema import GraphFact
from tests.conftest import (
    FakeEmbeddingProvider,
    FakeReranker,
    FakeVectorStore,
    settings_values,
)


class FakeCompletionProvider:
    async def complete(self, *, system_prompt: str, user_prompt: str) -> CompletionResult:
        del system_prompt, user_prompt
        return CompletionResult(text="Managers approve expenses above $500 [1].")


class FakeGraphSearch:
    async def search(self, question: str) -> list[GraphFact]:
        del question
        return []


class _Result:
    def __init__(self, value: float) -> None:
        self.value = value


class FakeMetric:
    def __init__(self, value: float) -> None:
        self._value = value

    async def ascore(self, **kwargs: Any) -> _Result:
        del kwargs
        return _Result(self._value)


def _settings() -> Settings:
    return Settings(_env_file=None, **settings_values())


def _bundle() -> PipelineBundle:
    return PipelineBundle(
        embedder=cast(Any, FakeEmbeddingProvider()),
        vector_store=cast(Any, FakeVectorStore()),
        reranker=cast(Any, FakeReranker()),
        synthesizer=AnswerSynthesizer(cast(Any, FakeCompletionProvider())),
        graph_search=cast(Any, FakeGraphSearch()),
    )


def _evaluator() -> RagasEvaluator:
    metrics = MetricBundle(
        faithfulness=cast(Any, FakeMetric(0.75)),
        context_precision=cast(Any, FakeMetric(0.65)),
        context_recall=cast(Any, FakeMetric(0.55)),
        answer_relevancy=cast(Any, FakeMetric(0.45)),
    )
    return RagasEvaluator(metrics, judge_model="sarvam-105b")


@pytest.mark.asyncio
async def test_committed_golden_set_runs_through_harness_and_writes_report(tmp_path: Path) -> None:
    settings = _settings()
    # Use a small slice of the real committed golden set to keep the test fast.
    full = load_golden_set(Path("data/eval/golden_set.jsonl"))
    golden = GoldenSet(items=full.items[:4])

    report = await run_ablation(settings, golden, _evaluator(), _bundle())

    assert [stage.stage for stage in report.stages] == list(ALL_STAGES)
    for stage in report.stages:
        scores = stage.result.aggregate
        assert None not in (
            scores.faithfulness,
            scores.context_precision,
            scores.context_recall,
            scores.answer_relevancy,
        )

    json_path, markdown_path = write_report(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload["stages"]) == 4

    markdown = markdown_path.read_text(encoding="utf-8")
    stage_rows = [line for line in markdown.splitlines() if line.startswith("| ") and "|" in line]
    # 4 stage data rows plus header + separator in the metrics table.
    assert any("naive (dense-only)" in row for row in stage_rows)
    assert any("+graph+multimodal" in row for row in stage_rows)
