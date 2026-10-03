"""Eval report writer: JSON + a markdown ablation table (Phase 6).

Writes the ablation outcome to the eval output directory as both a machine
readable JSON document (round-trips back into :class:`AblationReport`) and a
human-readable markdown table with one row per stage, each metric column, and a
per-stage delta. Any recorded judge failures are listed in a trailing section so
degraded scores are visible rather than hidden. The JSON write is atomic
(temp-then-rename), matching ``scripts/run_ingestion.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.evaluation.ablation import AblationReport, StageResult
from src.evaluation.ragas_eval import MetricScores

_STAGE_LABELS: dict[str, str] = {
    "naive": "naive (dense-only)",
    "hybrid_rerank": "+hybrid+rerank",
    "graph": "+graph",
    "graph_multimodal": "+graph+multimodal",
}
_METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("faithfulness", "Faithfulness"),
    ("context_precision", "Context Precision"),
    ("context_recall", "Context Recall"),
    ("answer_relevancy", "Answer Relevancy"),
)


def render_markdown_table(report: AblationReport) -> str:
    """Render the full markdown report as a string (pure, filesystem-free)."""
    lines: list[str] = [
        "# RAGAS Ablation Report",
        "",
        f"- Judge model: `{report.judge_model}`",
        f"- Golden items: {report.item_count}",
        f"- Corpus note: {report.corpus_notes}",
        "",
        "## Metrics by stage",
        "",
    ]
    header = ["Stage", *[label for _, label in _METRIC_COLUMNS], "Delta (vs prev)"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for stage in report.stages:
        lines.append("| " + " | ".join(_stage_row(stage)) + " |")

    failures = [failure for stage in report.stages for failure in stage.result.failures]
    lines.extend(["", "## Failures", ""])
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("None recorded.")
    lines.append("")
    return "\n".join(lines)


def write_report(report: AblationReport, output_dir: Path) -> tuple[Path, Path]:
    """Write the report as JSON and markdown; return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"eval_report_{timestamp}.json"
    markdown_path = output_dir / f"eval_report_{timestamp}.md"

    _atomic_write(
        json_path,
        json.dumps(report.model_dump(mode="json"), indent=2),
    )
    _atomic_write(markdown_path, render_markdown_table(report))
    return json_path, markdown_path


def _stage_row(stage: StageResult) -> list[str]:
    label = _STAGE_LABELS.get(stage.stage, stage.stage)
    cells = [label]
    cells.extend(_fmt(getattr(stage.result.aggregate, key)) for key, _ in _METRIC_COLUMNS)
    cells.append(_delta_cell(stage.delta))
    return cells


def _delta_cell(delta: MetricScores) -> str:
    parts = []
    for key, _ in _METRIC_COLUMNS:
        value = getattr(delta, key)
        if value is not None:
            parts.append(f"{value:+.3f}")
    return ", ".join(parts) if parts else "-"


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
