"""Tests for the eval report writer (JSON + markdown table)."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.ablation import AblationReport, StageResult
from src.evaluation.ragas_eval import EvalResult, ItemScore, MetricScores
from src.evaluation.report import render_markdown_table, write_report


def _stage(name: str, faithfulness: float, *, failures: list[str] | None = None) -> StageResult:
    scores = MetricScores(
        faithfulness=faithfulness,
        context_precision=0.8,
        context_recall=0.7,
        answer_relevancy=0.6,
    )
    return StageResult(
        stage=name,  # type: ignore[arg-type]
        result=EvalResult(
            per_item=[
                ItemScore(id="item-0", category="general", scores=scores),
            ],
            aggregate=scores,
            judge_model="sarvam-105b",
            failures=failures or [],
        ),
        delta=MetricScores(),
    )


def _report(*, failures: list[str] | None = None) -> AblationReport:
    return AblationReport(
        stages=[
            _stage("naive", 0.5),
            _stage("hybrid_rerank", 0.6),
            _stage("graph", 0.7, failures=failures),
            _stage("graph_multimodal", 0.7),
        ],
        judge_model="sarvam-105b",
        corpus_notes="Test corpus note.",
        item_count=1,
    )


def _table_rows(markdown: str) -> list[str]:
    """Return the data rows of the first pipe-delimited table in the markdown."""
    lines = [line for line in markdown.splitlines() if line.strip().startswith("|")]
    # lines[0] header, lines[1] separator, the rest are data rows.
    return lines[2:]


def test_markdown_has_one_row_per_stage() -> None:
    markdown = render_markdown_table(_report())

    rows = _table_rows(markdown)
    assert len(rows) == 4
    # Each row has: Stage + 4 metrics + Delta = 6 columns (7 pipe boundaries).
    for row in rows:
        assert row.count("|") == 7
    assert "naive" in rows[0]
    assert "+graph+multimodal" in rows[3]


def test_markdown_lists_failures_section() -> None:
    markdown_none = render_markdown_table(_report())
    assert "None recorded." in markdown_none

    markdown_fail = render_markdown_table(_report(failures=["item-0: faithfulness failed"]))
    assert "## Failures" in markdown_fail
    assert "item-0: faithfulness failed" in markdown_fail


def test_markdown_header_includes_judge_and_corpus_note() -> None:
    markdown = render_markdown_table(_report())

    assert "sarvam-105b" in markdown
    assert "Test corpus note." in markdown


def test_write_report_emits_parseable_json_and_markdown(tmp_path: Path) -> None:
    report = _report()

    json_path, markdown_path = write_report(report, tmp_path)

    assert json_path.exists()
    assert markdown_path.exists()

    # JSON round-trips back into the model.
    restored = AblationReport.model_validate(json.loads(json_path.read_text(encoding="utf-8")))
    assert restored.item_count == report.item_count
    assert [stage.stage for stage in restored.stages] == [
        "naive",
        "hybrid_rerank",
        "graph",
        "graph_multimodal",
    ]

    # Markdown has a header, separator, and one data row per stage.
    markdown = markdown_path.read_text(encoding="utf-8")
    assert len(_table_rows(markdown)) == 4


def test_write_report_creates_missing_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "reports"

    json_path, markdown_path = write_report(_report(), output_dir)

    assert json_path.parent == output_dir
    assert markdown_path.exists()
