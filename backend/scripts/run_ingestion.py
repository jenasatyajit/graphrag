"""GPU-VM batch ingestion command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, date, datetime
from pathlib import Path

import structlog

from src.api.config import Settings
from src.api.logging import configure_logging
from src.api.schemas import IngestionSummary
from src.indexing.embeddings import InProcessBGEM3
from src.indexing.vector_store import VectorStore
from src.ingestion.chunker import DocumentChunker
from src.ingestion.parser import DocType, DocumentParser

logger = structlog.get_logger(__name__)


async def run_ingestion(
    settings: Settings,
    *,
    doc_type: DocType,
    sop_version: str | None = None,
    effective_date: date | None = None,
) -> IngestionSummary:
    """Parse, chunk, embed, and index every supported file in ``data/raw``."""
    parser = DocumentParser()
    chunker = DocumentChunker()
    embedder = InProcessBGEM3(
        settings.embedding_model,
        device=settings.embedding_device,
        use_fp16=settings.embedding_use_fp16,
        expected_dimension=settings.embedding_dimension,
        inference_timeout_seconds=settings.embedding_inference_timeout_seconds,
    )
    store = VectorStore(settings)
    docs_processed = 0
    chunks_created = 0
    failures: list[str] = []

    try:
        await store.connect()
        sources = sorted(
            path
            for path in settings.ingestion_data_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".pdf", ".docx"}
        )
        for source in sources:
            try:
                document = parser.parse(
                    source,
                    doc_type=doc_type,
                    sop_version=sop_version if doc_type == "SOP" else None,
                    effective_date=effective_date,
                )
                chunks = chunker.chunk(document)
                for offset in range(0, len(chunks), settings.embedding_batch_size):
                    batch = chunks[offset : offset + settings.embedding_batch_size]
                    embeddings = await embedder.embed(
                        [chunk.text for chunk in batch],
                        mode="passage",
                    )
                    chunks_created += await store.upsert(batch, embeddings)
                docs_processed += 1
                logger.info(
                    "document_ingested", doc_id=str(document.doc_id), chunk_count=len(chunks)
                )
            except Exception:
                failures.append(source.name)
                logger.exception("document_ingestion_failed", source_name=source.name)
    finally:
        await store.close()

    summary = IngestionSummary(
        docs_processed=docs_processed,
        chunks_created=chunks_created,
        graph_nodes_created=0,
        graph_edges_created=0,
        failed_documents=failures,
        completed_at=datetime.now(UTC),
    )
    _write_summary(settings.ingestion_summary_path, summary)
    return summary


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
