"""Four-stage retrieval ablation runner (Phase 6, SPEC 7.6, BACKEND T4).

Runs the committed golden set through four pipeline configurations, scoring each
stage with the RAGAS metrics so the per-stage deltas show what each capability
adds:

1. ``naive``            - dense-only vector retrieval (``NaiveRagService``).
2. ``hybrid_rerank``    - hybrid search + rerank, graph disabled.
3. ``graph``            - hybrid + rerank + graph facts.
4. ``graph_multimodal`` - identical wiring to ``graph``; it differs only in that
   the ingested corpus contains table / image_caption chunks. There is no
   query-time switch that turns multimodal off, so stage 4 is a corpus
   distinction documented in ``corpus_notes`` rather than a code path.

Each stage composes the SAME real components as ``src/api/main.py:create_app``
(embedder, vector store, reranker, graph search, synthesizer) via a
:class:`PipelineBundle`, so the ablation measures the production pipeline, not a
re-implementation. Retrieved contexts are captured straight from the retrieval
step (``NaiveRagService.retrieve`` / ``HybridRagService.retrieve``) to avoid a
second retrieval round-trip per item.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import structlog
from pydantic import BaseModel, Field

from src.agent.router import HybridRagService, NaiveRagService
from src.api.config import Settings
from src.evaluation.golden_set import GoldenItem, GoldenSet
from src.evaluation.ragas_eval import EvalResult, MetricScores, PipelineAnswer, RagasEvaluator
from src.generation.synthesizer import AnswerSynthesizer
from src.indexing.embeddings import EmbeddingProvider
from src.indexing.vector_store import RetrievedChunk, VectorStore
from src.retrieval.graph_search import GraphSearchService
from src.retrieval.hybrid_search import HybridSearchService
from src.retrieval.reranker import Reranker

logger = structlog.get_logger(__name__)

AblationStage = Literal["naive", "hybrid_rerank", "graph", "graph_multimodal"]
ALL_STAGES: tuple[AblationStage, ...] = ("naive", "hybrid_rerank", "graph", "graph_multimodal")


@dataclass
class PipelineBundle:
    """Shared real components the ablation stages are composed from."""

    embedder: EmbeddingProvider
    vector_store: VectorStore
    reranker: Reranker
    synthesizer: AnswerSynthesizer
    graph_search: GraphSearchService


class StageResult(BaseModel):
    """One stage's RAGAS result and its metric deltas vs the previous stage."""

    stage: AblationStage
    result: EvalResult
    delta: MetricScores


class AblationReport(BaseModel):
    """The complete four-stage ablation outcome written to the eval report."""

    stages: list[StageResult]
    judge_model: str
    corpus_notes: str
    item_count: int = Field(ge=0)


def build_pipeline_bundle(settings: Settings) -> PipelineBundle:
    """Compose the real production components from ``Settings``.

    Mirrors ``create_app`` wiring: an HTTP embedding client, the Qdrant vector
    store, the OpenRouter reranker, a Sarvam-backed graph search, and the
    citation-grounding synthesizer. Imported lazily inside the function so unit
    tests can build a :class:`PipelineBundle` from fakes without importing vendor
    clients.
    """
    from src.generation.synthesizer import SarvamCompletionProvider
    from src.indexing.embeddings import HttpEmbeddingClient
    from src.indexing.graph_store import GraphStore
    from src.retrieval.reranker import OpenRouterReranker

    if settings.embedding_service_url is None:
        raise ValueError(
            "EMBEDDING_SERVICE_URL is not configured; the eval pipeline needs the "
            "query embedding service to retrieve contexts."
        )
    token = settings.embedding_service_token
    embedder = HttpEmbeddingClient(
        str(settings.embedding_service_url),
        token=token.get_secret_value() if token is not None else None,
        timeout_seconds=settings.embedding_request_timeout_seconds,
        max_retries=settings.embedding_max_retries,
        expected_dimension=settings.embedding_dimension,
    )
    completion = SarvamCompletionProvider(settings)
    extractor = SarvamCompletionProvider(
        settings,
        reasoning_effort=settings.llm_extract_reasoning_effort,
        max_tokens=settings.llm_extract_max_tokens,
    )
    graph_search = GraphSearchService(GraphStore(settings), extractor)
    return PipelineBundle(
        embedder=embedder,
        vector_store=VectorStore(settings),
        reranker=OpenRouterReranker(settings),
        synthesizer=AnswerSynthesizer(completion),
        graph_search=graph_search,
    )


def _naive_service(bundle: PipelineBundle) -> NaiveRagService:
    return NaiveRagService(bundle.embedder, bundle.vector_store, bundle.synthesizer)


def _hybrid_service(bundle: PipelineBundle, *, with_graph: bool) -> HybridRagService:
    return HybridRagService(
        HybridSearchService(bundle.embedder, bundle.vector_store),
        bundle.reranker,
        bundle.synthesizer,
        graph_search=bundle.graph_search if with_graph else None,
    )


async def _answer_naive(
    service: NaiveRagService, synthesizer: AnswerSynthesizer, question: str
) -> PipelineAnswer:
    chunks = await service.retrieve(question)
    grounded = await synthesizer.synthesize(question, chunks)
    return PipelineAnswer(answer=grounded.answer, retrieved_contexts=_chunk_texts(chunks))


async def _answer_hybrid(
    service: HybridRagService, synthesizer: AnswerSynthesizer, question: str
) -> PipelineAnswer:
    chunks, facts = await service.retrieve(question)
    grounded = await synthesizer.synthesize(question, chunks, graph_facts=facts)
    contexts = _chunk_texts(chunks) + [fact.as_text() for fact in facts]
    return PipelineAnswer(answer=grounded.answer, retrieved_contexts=contexts)


def _chunk_texts(chunks: list[RetrievedChunk]) -> list[str]:
    return [chunk.chunk.text for chunk in chunks]


async def _run_stage(
    stage: AblationStage,
    bundle: PipelineBundle,
    golden: GoldenSet,
) -> dict[str, PipelineAnswer]:
    """Produce a pipeline answer for every golden item under one stage."""
    answers: dict[str, PipelineAnswer] = {}
    for item in golden.items:
        try:
            answers[item.id] = await _answer_for_stage(stage, bundle, item)
        except Exception:
            # A per-item pipeline failure (retrieval/synthesis) must not abort the
            # stage; the evaluator records the missing answer as a failure.
            logger.warning("ablation_item_failed", stage=stage, item_id=item.id, exc_info=True)
    return answers


async def _answer_for_stage(
    stage: AblationStage, bundle: PipelineBundle, item: GoldenItem
) -> PipelineAnswer:
    if stage == "naive":
        return await _answer_naive(_naive_service(bundle), bundle.synthesizer, item.question)
    service = _hybrid_service(bundle, with_graph=stage in ("graph", "graph_multimodal"))
    return await _answer_hybrid(service, bundle.synthesizer, item.question)


async def run_ablation(
    settings: Settings,
    golden: GoldenSet,
    evaluator: RagasEvaluator,
    bundle: PipelineBundle,
    *,
    stages: tuple[AblationStage, ...] = ALL_STAGES,
) -> AblationReport:
    """Run the selected stages through the golden set and score each with RAGAS."""
    stage_results: list[StageResult] = []
    previous: MetricScores | None = None
    for stage in stages:
        logger.info("ablation_stage_started", stage=stage, item_count=len(golden.items))
        answers = await _run_stage(stage, bundle, golden)
        result = await evaluator.evaluate(golden.items, answers)
        delta = _delta(previous, result.aggregate)
        stage_results.append(StageResult(stage=stage, result=result, delta=delta))
        previous = result.aggregate
        logger.info(
            "ablation_stage_completed",
            stage=stage,
            scored_items=len(result.per_item),
            failures=len(result.failures),
        )
    return AblationReport(
        stages=stage_results,
        judge_model=evaluator.judge_model,
        corpus_notes=_corpus_notes(settings, stages),
        item_count=len(golden.items),
    )


def _delta(previous: MetricScores | None, current: MetricScores) -> MetricScores:
    """Per-metric change vs the previous stage (``None`` where either is absent)."""
    if previous is None:
        return MetricScores()
    return MetricScores(
        faithfulness=_sub(current.faithfulness, previous.faithfulness),
        context_precision=_sub(current.context_precision, previous.context_precision),
        context_recall=_sub(current.context_recall, previous.context_recall),
        answer_relevancy=_sub(current.answer_relevancy, previous.answer_relevancy),
    )


def _sub(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return current - previous


def _corpus_notes(settings: Settings, stages: tuple[AblationStage, ...]) -> str:
    """Explain honestly what the +graph+multimodal stage measures for this corpus."""
    if "graph_multimodal" not in stages:
        return "The +graph+multimodal stage was not run."
    if settings.eval_corpus_includes_multimodal:
        return (
            "The +graph+multimodal stage uses the same query path as +graph; it "
            "measures the gain from a corpus that INCLUDES table and image_caption "
            "chunks (EVAL_CORPUS_INCLUDES_MULTIMODAL=true)."
        )
    return (
        "The +graph+multimodal stage uses the same query path as +graph. "
        "EVAL_CORPUS_INCLUDES_MULTIMODAL is false, so the ingested corpus has no "
        "table/image_caption chunks and this stage is expected to match +graph; "
        "re-run after ingesting multimodal content to measure its contribution."
    )
