"""Docling-backed document parsing for text, tables, and pictures.

Phase 1 handled narrative text only. Phase 4 extends the parser to also emit
table blocks (exported to markdown) and picture blocks (captioned by a hosted
VLM through Docling's picture-description pipeline). Docling and its GPU deps
are imported lazily so the CPU query API never pulls them in.
"""

from __future__ import annotations

import contextlib
import importlib
import re
import unicodedata
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.api.errors import IngestionError

if TYPE_CHECKING:
    from src.api.config import Settings

DocType = Literal["SOP", "product_info"]

SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".pdf", ".docx", ".md"})

# Matches a trailing version marker on a filename stem (e.g. ``_v2``, ``-V3``,
# `` v3.0``, ``_v2.1``): a separator, a ``v``, then digits optionally followed by
# a dotted sub-version. Anchored at end-of-string and kept conservative so a
# non-version token like ``atlas_5000`` or ``model_v`` is never stripped.
_VERSION_SUFFIX = re.compile(r"[ _-]v\d+(?:\.\d+)?$", re.IGNORECASE)


def _title_from_stem(stem: str) -> str:
    """Derive a logical title from a filename stem, dropping a version suffix.

    Sibling version files (``expense_approval_sop_v2`` and
    ``expense_approval_sop_v3``) must collapse to the same ``doc_title`` so the
    SOP ``SUPERSEDES`` chain can match them. Only a trailing ``v``+number token
    preceded by a separator is removed; stems without such a marker are returned
    unchanged.
    """
    return _VERSION_SUFFIX.sub("", stem)


class ParsedTextBlock(BaseModel):
    """A narrative text item with its source-page provenance."""

    text: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    label: str | None = None


class ParsedTableBlock(BaseModel):
    """A table exported to markdown with its source-page provenance."""

    markdown: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)


class ParsedImageBlock(BaseModel):
    """A picture/chart caption plus a reference to the saved source image.

    ``caption`` is the VLM-generated description used for retrieval. ``image_path``
    points at the extracted image file for optional UI display, and stays ``None``
    when image bytes were unavailable. ``chart_markdown`` carries chart data that
    Docling's chart extraction converted to a table, when present.
    """

    caption: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    image_path: str | None = None
    chart_markdown: str | None = None


class ParsedDocument(BaseModel):
    """Typed output from the Docling parser (text, tables, and pictures)."""

    doc_id: UUID = Field(default_factory=uuid4)
    doc_title: str = Field(min_length=1)
    doc_type: DocType
    source_path: Path
    blocks: list[ParsedTextBlock]
    tables: list[ParsedTableBlock] = Field(default_factory=list)
    images: list[ParsedImageBlock] = Field(default_factory=list)
    sop_version: str | None = None
    effective_date: date | None = None


class _Converter(Protocol):
    def convert(self, source: Path) -> Any: ...


def _build_pdf_pipeline_options(settings: Settings) -> Any:
    """Configure Docling's PDF pipeline for table + picture extraction.

    Picture captioning is wired to a hosted OpenAI-compatible VLM (SPEC D-3)
    and only enabled when ``docling_vlm_url`` is configured; without it the
    pipeline still extracts tables and text, and simply skips captions.
    """
    pipeline_module = importlib.import_module("docling.datamodel.pipeline_options")
    pipeline_options = pipeline_module.PdfPipelineOptions()
    pipeline_options.do_table_structure = settings.docling_extract_tables
    pipeline_options.images_scale = settings.docling_image_scale
    # Retain picture image bytes so PictureItem.get_image() can save the figure.
    pipeline_options.generate_picture_images = True

    if settings.docling_vlm_url is not None:
        pipeline_options.enable_remote_services = True
        pipeline_options.do_picture_description = True
        pipeline_options.do_chart_extraction = settings.docling_extract_charts
        headers: dict[str, str] = {}
        if settings.docling_vlm_token is not None:
            headers["Authorization"] = f"Bearer {settings.docling_vlm_token.get_secret_value()}"
        params: dict[str, Any] = {}
        if settings.docling_vlm_model is not None:
            params["model"] = settings.docling_vlm_model
        pipeline_options.picture_description_options = pipeline_module.PictureDescriptionApiOptions(
            url=str(settings.docling_vlm_url),
            headers=headers,
            params=params,
            prompt=settings.docling_caption_prompt,
            timeout=settings.docling_caption_timeout_seconds,
        )
    return pipeline_options


def _default_converter_factory(settings: Settings) -> Callable[[], _Converter]:
    """Build a converter factory bound to the configured pipeline options.

    Keep Docling out of the CPU query API dependency closure. These imports are
    reached only by the GPU-hosted ingestion process.
    """

    def factory() -> _Converter:
        converter_module = importlib.import_module("docling.document_converter")
        base_models = importlib.import_module("docling.datamodel.base_models")
        pipeline_options = _build_pdf_pipeline_options(settings)
        pdf_format_option = converter_module.PdfFormatOption(pipeline_options=pipeline_options)
        return converter_module.DocumentConverter(
            format_options={base_models.InputFormat.PDF: pdf_format_option}
        )

    return factory


def _default_converter() -> _Converter:
    # Backwards-compatible bare converter (no multimodal pipeline options). Used
    # only when a DocumentParser is constructed without settings.
    module = importlib.import_module("docling.document_converter")
    converter_class = module.DocumentConverter
    return converter_class()


class DocumentParser:
    """Convert PDF/DOCX/Markdown documents into text, table, and picture blocks.

    When constructed with ``settings``, Docling is configured to extract table
    structure and to caption pictures/charts through the hosted VLM. Markdown has
    no page layout, so ``page_number`` stays ``None`` for those chunks. A
    converter can be injected so unit tests never initialize Docling or download
    models. ``image_output_dir`` controls where extracted picture images are
    written; it defaults to the parsed document's directory when unset.
    """

    _SUPPORTED_SUFFIXES = SUPPORTED_SUFFIXES

    def __init__(
        self,
        converter_factory: Callable[[], _Converter] | None = None,
        *,
        settings: Settings | None = None,
        image_output_dir: Path | None = None,
    ) -> None:
        if converter_factory is None:
            converter_factory = (
                _default_converter_factory(settings) if settings is not None else _default_converter
            )
        self._converter_factory = converter_factory
        self._image_output_dir = image_output_dir
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
        """Parse one supported document into text, table, and picture blocks."""
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

        resolved_doc_id = doc_id or uuid4()
        try:
            converter = self._converter
            if converter is None:
                converter = self._converter_factory()
                self._converter = converter
            result = converter.convert(path)
            document = result.document
            text_items = document.texts
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

        image_dir = self._image_dir_for(path, resolved_doc_id)
        tables = _extract_tables(document)
        images = _extract_images(document, image_dir=image_dir)

        return ParsedDocument(
            doc_id=resolved_doc_id,
            doc_title=(doc_title or _title_from_stem(path.stem)).strip(),
            doc_type=doc_type,
            source_path=path,
            blocks=blocks,
            tables=tables,
            images=images,
            sop_version=sop_version,
            effective_date=effective_date,
        )

    def _image_dir_for(self, path: Path, doc_id: UUID) -> Path:
        base = self._image_output_dir or path.parent
        return base / str(doc_id)


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


def _page_number(item: Any) -> int | None:
    provenance = getattr(item, "prov", None) or []
    return getattr(provenance[0], "page_no", None) if provenance else None


def _extract_tables(document: Any) -> list[ParsedTableBlock]:
    """Export every detected table to markdown, dropping empty ones."""
    tables: list[ParsedTableBlock] = []
    for table in getattr(document, "tables", None) or []:
        try:
            markdown = _normalize_text(str(table.export_to_markdown(document))).strip()
        except Exception:
            markdown = ""
        if not markdown:
            continue
        tables.append(ParsedTableBlock(markdown=markdown, page_number=_page_number(table)))
    return tables


def _extract_images(document: Any, *, image_dir: Path) -> list[ParsedImageBlock]:
    """Collect VLM captions and saved image references for each picture.

    Captions are produced by Docling's picture-description pipeline during
    conversion and read back off each ``PictureItem``. Pictures without a caption
    (e.g. when the VLM is not configured) are skipped so no empty caption chunk is
    created. Chart data extracted by Docling is surfaced as ``chart_markdown``.
    """
    images: list[ParsedImageBlock] = []
    for index, picture in enumerate(getattr(document, "pictures", None) or []):
        caption = _picture_caption(picture, document)
        if not caption:
            continue
        image_path = _save_picture_image(picture, document, image_dir=image_dir, index=index)
        images.append(
            ParsedImageBlock(
                caption=caption,
                page_number=_page_number(picture),
                image_path=image_path,
                chart_markdown=_picture_chart_markdown(picture, document),
            )
        )
    return images


def _picture_caption(picture: Any, document: Any) -> str | None:
    """Return the VLM description for a picture, if Docling produced one.

    Docling exposes picture descriptions through ``caption_text`` and, on older
    document models, through per-item ``annotations``. Try both so the caption is
    found regardless of the docling-core version.
    """
    caption_getter = getattr(picture, "caption_text", None)
    if callable(caption_getter):
        with contextlib.suppress(Exception):
            caption = _normalize_text(str(caption_getter(document) or "")).strip()
            if caption:
                return caption
    for annotation in getattr(picture, "annotations", None) or []:
        text = getattr(annotation, "text", None)
        if text:
            caption = _normalize_text(str(text)).strip()
            if caption:
                return caption
    return None


def _picture_chart_markdown(picture: Any, document: Any) -> str | None:
    """Return chart data converted to a markdown table, when Docling extracted it."""
    for annotation in getattr(picture, "annotations", None) or []:
        chart = getattr(annotation, "chart_data", None) or getattr(annotation, "table", None)
        exporter = getattr(chart, "export_to_markdown", None)
        if callable(exporter):
            with contextlib.suppress(Exception):
                markdown = _normalize_text(str(exporter(document))).strip()
                if markdown:
                    return markdown
    return None


def _save_picture_image(
    picture: Any,
    document: Any,
    *,
    image_dir: Path,
    index: int,
) -> str | None:
    """Persist a picture's image bytes for optional UI display.

    Returns the saved file path as a string, or ``None`` when image bytes are
    unavailable or cannot be written. A failure to save an image never aborts
    ingestion — the caption alone still supports retrieval.
    """
    getter = getattr(picture, "get_image", None)
    if not callable(getter):
        return None
    try:
        image = getter(document)
        if image is None:
            return None
        image_dir.mkdir(parents=True, exist_ok=True)
        target = image_dir / f"picture-{index + 1}.png"
        image.save(target, "PNG")
        return str(target)
    except Exception:
        return None
