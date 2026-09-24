"""SOP-aware chunking with a bounded prose fallback."""

from __future__ import annotations

import re
from datetime import date
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.ingestion.parser import ParsedDocument, ParsedImageBlock, ParsedTextBlock

ContentType = Literal["text", "table", "image_caption"]
SourceTier = Literal["internal"]

_EXPLICIT_STEP_RE = re.compile(
    r"^\s*step\s+(\d+)\s*(?:[:.\-)\u2014\u2013]\s*|\s+)(.*)$", re.IGNORECASE
)
_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d+)\s*[.\-)]\s+(.+)$")
_HEADING_LABELS = frozenset({"section_header", "title", "list_item"})


class Chunk(BaseModel):
    """Qdrant payload contract for one ingested internal-document chunk."""

    chunk_id: UUID = Field(default_factory=uuid4)
    doc_id: UUID
    doc_title: str = Field(min_length=1)
    doc_type: Literal["SOP", "product_info"]
    content_type: ContentType = "text"
    text: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    sop_version: str | None = None
    effective_date: date | None = None
    is_current_version: bool = True
    entity_ids: list[str] = Field(default_factory=list)
    source_tier: SourceTier = "internal"
    step_order: int | None = Field(default=None, ge=1)
    # Path to the original extracted image, set only for image_caption chunks so
    # the UI can optionally display the source figure alongside its caption.
    image_path: str | None = None


class DocumentChunker:
    """Create step-level SOP chunks and section/prose chunks for other text."""

    def __init__(self, *, max_chars: int = 1600, overlap_chars: int = 160) -> None:
        if max_chars < 100:
            raise ValueError("max_chars must be at least 100")
        if overlap_chars < 0 or overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be non-negative and smaller than max_chars")
        self._max_chars = max_chars
        self._overlap_chars = overlap_chars

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Chunk a parsed document's narrative text without losing metadata.

        This covers text only. Multimodal table and image-caption chunks are
        produced by ``chunk_tables`` and ``chunk_images`` so callers can route
        them differently (e.g. skip graph extraction for captions).
        """
        if document.doc_type == "SOP":
            segments = self._sop_segments(document.blocks)
        else:
            segments = self._product_segments(document.blocks)

        chunks: list[Chunk] = []
        for text, page_number, step_order in segments:
            parts = [text] if step_order is not None else self._split_text(text)
            for part in parts:
                chunks.append(
                    self._build_chunk(
                        document,
                        text=part,
                        page_number=page_number,
                        step_order=step_order,
                    )
                )
        return chunks

    def chunk_tables(self, document: ParsedDocument) -> list[Chunk]:
        """Build one ``table`` chunk per extracted table markdown block.

        Tables are kept whole (one chunk each) so row/column relationships and
        the surrounding markdown structure survive into retrieval intact.
        """
        chunks: list[Chunk] = []
        for table in document.tables:
            markdown = table.markdown.strip()
            if not markdown:
                continue
            chunks.append(
                self._build_chunk(
                    document,
                    text=markdown,
                    page_number=table.page_number,
                    step_order=None,
                    content_type="table",
                )
            )
        return chunks

    def chunk_images(self, document: ParsedDocument) -> list[Chunk]:
        """Build chunks for picture captions and any chart-derived table data.

        Each captioned picture yields one ``image_caption`` chunk carrying the
        original image reference. When Docling's chart extraction produced tabular
        data for that picture, an additional ``table`` chunk is emitted so the
        structured chart values are retrievable alongside the caption.
        """
        chunks: list[Chunk] = []
        for image in document.images:
            chunks.append(self._image_caption_chunk(document, image))
            chart_markdown = (image.chart_markdown or "").strip()
            if chart_markdown:
                chunks.append(
                    self._build_chunk(
                        document,
                        text=chart_markdown,
                        page_number=image.page_number,
                        step_order=None,
                        content_type="table",
                    )
                )
        return chunks

    def _image_caption_chunk(self, document: ParsedDocument, image: ParsedImageBlock) -> Chunk:
        return self._build_chunk(
            document,
            text=image.caption.strip(),
            page_number=image.page_number,
            step_order=None,
            content_type="image_caption",
            image_path=image.image_path,
        )

    def _sop_segments(
        self,
        blocks: list[ParsedTextBlock],
    ) -> list[tuple[str, int | None, int | None]]:
        segments: list[tuple[str, int | None, int | None]] = []
        pending_prose: list[ParsedTextBlock] = []
        current_step: list[ParsedTextBlock] = []
        current_order: int | None = None

        def flush_step() -> None:
            nonlocal current_step
            if current_step:
                segments.append(
                    (
                        "\n\n".join(block.text for block in current_step),
                        current_step[0].page_number,
                        current_order,
                    )
                )
                current_step = []

        for block in blocks:
            step_order = _step_order(block)
            if step_order is not None:
                if current_order is None and pending_prose:
                    segments.extend(_prose_segments(pending_prose))
                    pending_prose = []
                flush_step()
                current_order = step_order
                current_step = [block]
            elif current_order is not None:
                current_step.append(block)
            else:
                pending_prose.append(block)

        flush_step()
        segments.extend(_prose_segments(pending_prose))
        return segments

    def _product_segments(
        self,
        blocks: list[ParsedTextBlock],
    ) -> list[tuple[str, int | None, int | None]]:
        segments: list[tuple[str, int | None, int | None]] = []
        current: list[ParsedTextBlock] = []
        for block in blocks:
            if block.label in _HEADING_LABELS and current:
                segments.extend(_prose_segments(current))
                current = []
            current.append(block)
        segments.extend(_prose_segments(current))
        return segments

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self._max_chars:
            return [text]

        parts: list[str] = []
        start = 0
        while start < len(text):
            target_end = min(start + self._max_chars, len(text))
            end = target_end
            if target_end < len(text):
                boundary = text.rfind(" ", start, target_end)
                if boundary > start:
                    end = boundary
            part = text[start:end].strip()
            if part:
                parts.append(part)
            if end >= len(text):
                break
            next_start = max(0, end - self._overlap_chars)
            if next_start <= start:
                next_start = end
            start = next_start
        return parts

    @staticmethod
    def _build_chunk(
        document: ParsedDocument,
        *,
        text: str,
        page_number: int | None,
        step_order: int | None,
        content_type: ContentType = "text",
        image_path: str | None = None,
    ) -> Chunk:
        return Chunk(
            doc_id=document.doc_id,
            doc_title=document.doc_title,
            doc_type=document.doc_type,
            content_type=content_type,
            text=text,
            page_number=page_number,
            sop_version=document.sop_version,
            effective_date=document.effective_date,
            step_order=step_order,
            image_path=image_path,
        )


def _step_order(block: ParsedTextBlock) -> int | None:
    explicit_match = _EXPLICIT_STEP_RE.match(block.text)
    if explicit_match:
        return int(explicit_match.group(1))
    if block.label in _HEADING_LABELS and (heading_match := _NUMBERED_HEADING_RE.match(block.text)):
        return int(heading_match.group(1))
    return None


def _prose_segments(
    blocks: list[ParsedTextBlock],
) -> list[tuple[str, int | None, int | None]]:
    if not blocks:
        return []
    return [("\n\n".join(block.text for block in blocks), blocks[0].page_number, None)]
