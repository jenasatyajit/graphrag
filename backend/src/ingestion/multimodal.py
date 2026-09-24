"""Phase 4 multimodal ingestion orchestration.

Tables and pictures are extracted by Docling during parsing (see ``parser.py``):
tables are exported to markdown and pictures are captioned by a hosted VLM through
Docling's picture-description pipeline (SPEC D-3). This module is a thin
orchestrator that turns those parsed blocks into embeddable chunks and keeps the
routing rule for graph extraction in one place: table chunks carry structured
facts worth extracting, while image-caption chunks are prose descriptions that are
deliberately excluded from graph extraction to avoid noisy triples.
"""

from __future__ import annotations

from src.ingestion.chunker import Chunk, DocumentChunker
from src.ingestion.parser import ParsedDocument


class MultimodalChunkSet:
    """Multimodal chunks split by their downstream graph-extraction routing.

    ``graph_eligible`` chunks (tables, including chart-derived tables) are sent
    through graph extraction like narrative text. ``caption_only`` chunks
    (image captions) are embedded and indexed but skip graph extraction.
    """

    def __init__(self, graph_eligible: list[Chunk], caption_only: list[Chunk]) -> None:
        self.graph_eligible = graph_eligible
        self.caption_only = caption_only

    @property
    def all(self) -> list[Chunk]:
        """Every multimodal chunk, in a stable table-then-caption order."""
        return [*self.graph_eligible, *self.caption_only]

    @property
    def table_count(self) -> int:
        return len(self.graph_eligible)

    @property
    def image_count(self) -> int:
        return len(self.caption_only)


class MultimodalIngestor:
    """Build table and image-caption chunks from a parsed document."""

    def __init__(self, chunker: DocumentChunker | None = None) -> None:
        self._chunker = chunker or DocumentChunker()

    def build_chunks(self, document: ParsedDocument) -> MultimodalChunkSet:
        """Return multimodal chunks grouped by graph-extraction eligibility.

        Table chunks (from real tables and from chart extraction) are graph
        eligible; image-caption chunks are caption-only. Splitting them here lets
        the ingestion driver run graph extraction over text + tables while leaving
        captions out, without the driver needing to inspect ``content_type``.
        """
        table_chunks = self._chunker.chunk_tables(document)
        image_chunks = self._chunker.chunk_images(document)

        graph_eligible = [*table_chunks]
        caption_only: list[Chunk] = []
        for chunk in image_chunks:
            if chunk.content_type == "table":
                graph_eligible.append(chunk)
            else:
                caption_only.append(chunk)
        return MultimodalChunkSet(graph_eligible=graph_eligible, caption_only=caption_only)
