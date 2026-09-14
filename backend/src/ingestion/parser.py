"""Docling-backed, text-only document parsing for Phase 1 ingestion."""

from __future__ import annotations

import contextlib
import importlib
import unicodedata
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.api.errors import IngestionError

DocType = Literal["SOP", "product_info"]

SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".pdf", ".docx", ".md"})


class ParsedTextBlock(BaseModel):
    """A narrative text item with its source-page provenance."""

    text: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    label: str | None = None


class ParsedDocument(BaseModel):
    """Typed output from the text-only Docling parser."""

    doc_id: UUID = Field(default_factory=uuid4)
    doc_title: str = Field(min_length=1)
    doc_type: DocType
    source_path: Path
    blocks: list[ParsedTextBlock]
    sop_version: str | None = None
    effective_date: date | None = None


class _Converter(Protocol):
    def convert(self, source: Path) -> Any: ...


def _default_converter() -> _Converter:
    # Keep Docling out of the CPU query API dependency closure. This import is
    # reached only by the GPU-hosted ingestion process.
    module = importlib.import_module("docling.document_converter")
    converter_class = module.DocumentConverter
    return converter_class()


class DocumentParser:
    """Convert PDF/DOCX/Markdown documents into narrative blocks using Docling.

    Tables and pictures are deliberately omitted until Phase 4. Markdown has no
    page layout, so ``page_number`` stays ``None`` for those chunks. A converter
    can be injected so unit tests never initialize Docling or download models.
    """

    _SUPPORTED_SUFFIXES = SUPPORTED_SUFFIXES

    def __init__(self, converter_factory: Callable[[], _Converter] = _default_converter) -> None:
        self._converter_factory = converter_factory
        self._converter: _Converter | None = None

    def parse(
        self,
        source: Path,
        *,
        doc_type: DocType,
        doc_title: str | None = None,
        sop_version: str | None = None,
        effective_date: date | None = None,
        doc_id: UUID | None = None,
    ) -> ParsedDocument:
        """Parse one supported document while preserving text-item page numbers."""
        path = source.resolve()
        if not path.is_file():
            raise IngestionError(
                "The source document does not exist.", details={"path": str(source)}
            )
        if path.suffix.lower() not in self._SUPPORTED_SUFFIXES:
            raise IngestionError(
                "The source document format is not supported.",
                details={
                    "suffix": path.suffix.lower(),
                    "supported": sorted(self._SUPPORTED_SUFFIXES),
                },
            )

        try:
            converter = self._converter
            if converter is None:
                converter = self._converter_factory()
                self._converter = converter
            result = converter.convert(path)
            text_items = result.document.texts
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(
                "Docling could not parse the source document.",
                details={"path": str(source)},
            ) from exc

        blocks = [block for item in text_items if (block := _to_text_block(item)) is not None]
        if not blocks:
            raise IngestionError(
                "The source document contains no narrative text.",
                details={"path": str(source)},
            )

        return ParsedDocument(
            doc_id=doc_id or uuid4(),
            doc_title=(doc_title or path.stem).strip(),
            doc_type=doc_type,
            source_path=path,
            blocks=blocks,
            sop_version=sop_version,
            effective_date=effective_date,
        )


# Typographic characters that add no meaning for retrieval but produce noisy
# citations. Mapped to plain ASCII equivalents after Unicode normalization.
_PUNCTUATION_MAP = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
    "\u00a0": " ",
}


def _normalize_text(text: str) -> str:
    """Repair mojibake and normalize Unicode so chunks stay clean and citable.

    Docling output occasionally contains UTF-8 bytes that were previously decoded
    as Latin-1 (mojibake such as ``â€“``). Re-encoding through Latin-1 recovers the
    original characters; NFC normalization and a small punctuation map then keep
    stored chunks and citations readable.
    """
    if any(marker in text for marker in ("\u00c3", "\u00e2\u0080")):
        with contextlib.suppress(UnicodeEncodeError, UnicodeDecodeError):
            text = text.encode("latin-1").decode("utf-8")
    text = unicodedata.normalize("NFC", text)
    for source, target in _PUNCTUATION_MAP.items():
        text = text.replace(source, target)
    return text


def _to_text_block(item: Any) -> ParsedTextBlock | None:
    text = _normalize_text(str(getattr(item, "text", ""))).strip()
    if not text:
        return None

    provenance = getattr(item, "prov", None) or []
    page_number = getattr(provenance[0], "page_no", None) if provenance else None
    raw_label = getattr(item, "label", None)
    label_value = getattr(raw_label, "value", raw_label)
    return ParsedTextBlock(
        text=text,
        page_number=page_number,
        label=str(label_value) if label_value is not None else None,
    )
