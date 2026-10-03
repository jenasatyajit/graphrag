"""Evaluation and ablation components (Phase 6: RAGAS + ablation)."""

from __future__ import annotations

from src.evaluation.ablation import (
    ALL_STAGES,
    AblationReport,
    AblationStage,
    PipelineBundle,
    StageResult,
    build_pipeline_bundle,
    run_ablation,
)
from src.evaluation.golden_set import (
    REQUIRED_MINIMUMS,
    GoldenItem,
    GoldenSet,
    load_golden_set,
)
from src.evaluation.ragas_embeddings import HttpEmbeddingAdapter
from src.evaluation.ragas_eval import (
    EvalResult,
    ItemScore,
    MetricBundle,
    MetricScores,
    PipelineAnswer,
    RagasEvaluator,
    build_default_metrics,
    build_embeddings_adapter,
    build_sarvam_judge,
)
from src.evaluation.report import render_markdown_table, write_report

__all__ = [
    "ALL_STAGES",
    "REQUIRED_MINIMUMS",
    "AblationReport",
    "AblationStage",
    "EvalResult",
    "GoldenItem",
    "GoldenSet",
    "HttpEmbeddingAdapter",
    "ItemScore",
    "MetricBundle",
    "MetricScores",
    "PipelineAnswer",
    "PipelineBundle",
    "RagasEvaluator",
    "StageResult",
    "build_default_metrics",
    "build_embeddings_adapter",
    "build_pipeline_bundle",
    "build_sarvam_judge",
    "load_golden_set",
    "render_markdown_table",
    "run_ablation",
    "write_report",
]
