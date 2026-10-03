"""RAGAS + ablation evaluation command-line entry point (Phase 6).

Live offline evaluation harness. It is NOT part of the default ``uv run pytest``
run; it talks to the real cloud stores (Qdrant/Neo4j), the GPU embedding service,
and the Sarvam judge. Run it from ``backend/`` with a populated ``.env``:

    uv run python scripts/run_eval.py
    uv run python scripts/run_eval.py --stages naive,hybrid_rerank --limit 5
    uv run python scripts/run_eval.py --golden-set data/eval/golden_set.jsonl \
        --output-dir data/eval/reports

Required .env for a real run: SARVAM_API_KEY (judge), EMBEDDING_SERVICE_URL plus
QDRANT_* and NEO4J_* (retrieval and the stores must contain ingested chunks).
The script runs one minimal judge call first and aborts with a clear message if
auth/model is wrong, and it degrades (never crashes) on empty stores or a single
judge failure - those are logged and recorded in the report.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Make ``src`` importable regardless of the current working directory: add the
# backend/ root (this file's parent's parent) to sys.path before importing.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import structlog  # noqa: E402

from src.api.config import Settings  # noqa: E402
from src.api.logging import configure_logging  # noqa: E402
from src.evaluation.ablation import (  # noqa: E402
    ALL_STAGES,
    AblationStage,
    PipelineBundle,
    build_pipeline_bundle,
    run_ablation,
)
from src.evaluation.golden_set import GoldenSet, load_golden_set  # noqa: E402
from src.evaluation.ragas_eval import (  # noqa: E402
    RagasEvaluator,
    build_default_metrics,
    build_embeddings_adapter,
    build_sarvam_judge,
)
from src.evaluation.report import write_report  # noqa: E402

logger = structlog.get_logger(__name__)


def _parse_stages(raw: str) -> tuple[AblationStage, ...]:
    valid = set(ALL_STAGES)
    stages: list[AblationStage] = []
    for name in (part.strip() for part in raw.split(",")):
        if not name:
            continue
        if name not in valid:
            raise SystemExit(f"Unknown stage '{name}'. Valid stages: {', '.join(ALL_STAGES)}.")
        stages.append(name)  # type: ignore[arg-type]
    return tuple(stages) if stages else ALL_STAGES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS + the four-stage ablation study.")
    parser.add_argument(
        "--stages",
        default=",".join(ALL_STAGES),
        help="Comma-separated subset of stages to run (default: all four).",
    )
    parser.add_argument("--golden-set", type=Path, help="Golden set path override.")
    parser.add_argument("--output-dir", type=Path, help="Report output directory override.")
    parser.add_argument(
        "--limit",
        type=int,
        help="Cap the number of golden items (quick smoke run).",
    )
    return parser.parse_args()


async def _verify_judge(judge: object, settings: Settings) -> None:
    """Run one minimal judge call so an auth/model error aborts before the run.

    Uses the InstructorBaseRagasLLM ``agenerate`` surface the metrics rely on.
    """
    from pydantic import BaseModel

    class _Ping(BaseModel):
        ok: bool

    try:
        await judge.agenerate("Reply with ok=true.", _Ping)  # type: ignore[attr-defined]
    except Exception as exc:  # surface a clear, actionable message
        raise SystemExit(
            "The Sarvam judge rejected a minimal test call "
            f"(model={settings.eval_judge_model}, base_url={settings.eval_judge_base_url}): "
            f"{type(exc).__name__}: {exc}. Check SARVAM_API_KEY and EVAL_JUDGE_* settings."
        ) from exc


def _limit(golden: GoldenSet, limit: int | None) -> GoldenSet:
    if limit is None or limit >= len(golden.items):
        return golden
    return GoldenSet(items=golden.items[:limit])


async def _preflight_stores(bundle: PipelineBundle) -> None:
    """Fail fast with a clear message when the vector store is empty/unreachable."""
    health = await bundle.vector_store.verify_connectivity()
    if health.status != "ok":
        raise SystemExit(
            "The vector store is not ready for evaluation: "
            f"{health.detail or 'unavailable'}. Ingest documents before running eval."
        )


async def _run(args: argparse.Namespace) -> None:
    settings = Settings()
    configure_logging(settings.app_environment)

    golden_path = args.golden_set or settings.eval_golden_set_path
    output_dir = args.output_dir or settings.eval_output_dir
    stages = _parse_stages(args.stages)

    try:
        golden = _limit(load_golden_set(golden_path), args.limit)
    except ValueError as exc:
        raise SystemExit(f"Could not load the golden set: {exc}") from exc

    try:
        bundle = build_pipeline_bundle(settings)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    await _preflight_stores(bundle)

    judge = build_sarvam_judge(settings)
    await _verify_judge(judge, settings)
    embeddings = build_embeddings_adapter(bundle.embedder)
    metrics = build_default_metrics(judge, embeddings)
    evaluator = RagasEvaluator(
        metrics,
        judge_model=settings.eval_judge_model,
        max_concurrency=settings.eval_judge_max_concurrency,
    )

    logger.info("eval_run_started", stages=list(stages), item_count=len(golden.items))
    report = await run_ablation(settings, golden, evaluator, bundle, stages=stages)
    json_path, markdown_path = write_report(report, output_dir)

    total_failures = sum(len(stage.result.failures) for stage in report.stages)
    logger.info(
        "eval_run_completed",
        stages=len(report.stages),
        item_count=report.item_count,
        failures=total_failures,
        json_report=str(json_path),
        markdown_report=str(markdown_path),
    )
    print(f"Wrote eval report:\n  JSON:     {json_path}\n  Markdown: {markdown_path}")
    if total_failures:
        print(
            f"Recorded {total_failures} judge/metric failure(s); see the report's Failures section."
        )


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
