"""GPU-VM batch ingestion command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

# Make ``src`` importable regardless of the current working directory: add the
# backend/ root (this file's parent's parent) to sys.path before importing.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import structlog  # noqa: E402

from src.api.config import Settings  # noqa: E402
from src.api.logging import configure_logging  # noqa: E402
from src.api.schemas import IngestionSummary  # noqa: E402
from src.indexing.embeddings import InProcessBGEM3  # noqa: E402
from src.indexing.graph_store import GraphStore  # noqa: E402
from src.indexing.vector_store import VectorStore  # noqa: E402
from src.ingestion.chunker import Chunk, DocumentChunker  # noqa: E402
from src.ingestion.graph_extractor import GraphExtractor  # noqa: E402
from src.ingestion.multimodal import MultimodalIngestor  # noqa: E402
from src.ingestion.parser import SUPPORTED_SUFFIXES, DocType, DocumentParser  # noqa: E402

logger = structlog.get_logger(__name__)


async def run_ingestion(
    settings: Settings,
    *,
    doc_type: DocType,
    sop_version: str | None = None,
    effective_date: date | None = None,
) -> IngestionSummary:
    """Parse, chunk, embed, and index every supported file in ``data/raw``."""
    parser = DocumentParser(settings=settings, image_output_dir=settings.ingestion_image_dir)
    chunker = DocumentChunker()
    multimodal = MultimodalIngestor(chunker)
    embedder = InProcessBGEM3(
        settings.embedding_model,
        device=settings.embedding_device,
        use_fp16=settings.embedding_use_fp16,
        expected_dimension=settings.embedding_dimension,
        inference_timeout_seconds=settings.embedding_inference_timeout_seconds,
    )
    store = VectorStore(settings)
    graph_store = GraphStore(settings)
    extractor = GraphExtractor(settings)
    docs_processed = 0
    chunks_created = 0
    tables_extracted = 0
    images_extracted = 0
    graph_nodes: set[tuple[str, str]] = set()
    graph_edges = 0
    superseded_titles: list[tuple[str, str]] = []
    failures: list[str] = []

    try:
        await store.connect()
        await graph_store.connect()
        sources = sorted(
            path
            for path in settings.ingestion_data_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        for source in sources:
            try:
                document = parser.parse(
                    source,
                    doc_type=doc_type,
                    sop_version=sop_version if doc_type == "SOP" else None,
                    effective_date=effective_date,
                )
                text_chunks = chunker.chunk(document)
                multimodal_chunks = multimodal.build_chunks(document)
                tables_extracted += multimodal_chunks.table_count
                images_extracted += multimodal_chunks.image_count

                # Graph extraction runs over text and table chunks (structured
                # facts) but not image captions, which are prose and would add
                # noisy triples. All chunks are embedded and indexed.
                graph_chunks = [*text_chunks, *multimodal_chunks.graph_eligible]
                edges, nodes = await _extract_and_write_graph(extractor, graph_store, graph_chunks)
                graph_edges += edges
                graph_nodes |= nodes

                chunks = [*text_chunks, *multimodal_chunks.all]
                for offset in range(0, len(chunks), settings.embedding_batch_size):
                    batch = chunks[offset : offset + settings.embedding_batch_size]
                    embeddings = await embedder.embed(
                        [chunk.text for chunk in batch],
                        mode="passage",
                    )
                    chunks_created += await store.upsert(batch, embeddings)
                if doc_type == "SOP" and sop_version is not None:
                    superseded_titles.append((document.doc_title, sop_version))
                docs_processed += 1
                logger.info(
                    "document_ingested",
                    doc_id=str(document.doc_id),
                    chunk_count=len(chunks),
                    table_count=multimodal_chunks.table_count,
                    image_count=multimodal_chunks.image_count,
                    graph_edges=edges,
                )
            except Exception:
                failures.append(source.name)
                logger.exception("document_ingestion_failed", source_name=source.name)

        # Build SOP version chains after all docs are written, then reconcile
        # the vector store's is_current_version flags with the graph.
        for title, version in superseded_titles:
            try:
                await graph_store.supersede_previous_versions(title, version)
            except Exception:
                logger.exception("sop_supersession_failed", doc_title=title)
        try:
            outdated_chunk_ids = await graph_store.superseded_chunk_ids()
            if outdated_chunk_ids:
                await store.mark_superseded(sorted(outdated_chunk_ids))
        except Exception:
            logger.exception("version_flag_reconciliation_failed")
    finally:
        await store.close()
        await graph_store.close()

    summary = IngestionSummary(
        docs_processed=docs_processed,
        chunks_created=chunks_created,
        tables_extracted=tables_extracted,
        images_extracted=images_extracted,
        graph_nodes_created=len(graph_nodes),
        graph_edges_created=graph_edges,
        failed_documents=failures,
        completed_at=datetime.now(UTC),
    )
    _write_summary(settings.ingestion_summary_path, summary)
    return summary


async def _extract_and_write_graph(
    extractor: GraphExtractor,
    graph_store: GraphStore,
    chunks: list[Chunk],
) -> tuple[int, set[tuple[str, str]]]:
    """Extract triples per chunk, write them, and back-populate entity ids.

    Returns the edge count and the set of unique ``(label, name)`` nodes touched
    so the caller can report accurate graph counts. Mutates each chunk's
    ``entity_ids`` in place before it is embedded and upserted.
    """
    total_edges = 0
    nodes: set[tuple[str, str]] = set()
    for chunk in chunks:
        triples = await extractor.extract(chunk)
        if not triples:
            continue
        total_edges += await graph_store.write_triples(triples, chunk_id=str(chunk.chunk_id))
        entity_ids: list[str] = []
        for triple in triples:
            for node in (triple.subject, triple.object):
                key = (node.label, node.name)
                nodes.add(key)
                entity_ids.append(f"{node.label}:{node.name}")
        chunk.entity_ids = list(dict.fromkeys(entity_ids))
    return total_edges, nodes


def _write_summary(path: Path, summary: IngestionSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest internal documents on the GPU VM.")
    parser.add_argument("--doc-type", choices=("SOP", "product_info"), required=True)
    parser.add_argument("--sop-version")
    parser.add_argument("--effective-date", type=date.fromisoformat)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = Settings()
    configure_logging(settings.app_environment)
    summary = asyncio.run(
        run_ingestion(
            settings,
            doc_type=args.doc_type,
            sop_version=args.sop_version,
            effective_date=args.effective_date,
        )
    )
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
