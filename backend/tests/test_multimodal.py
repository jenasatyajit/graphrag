"""Phase 4 multimodal chunk orchestration: table/caption routing for the graph."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.multimodal import MultimodalIngestor
from src.ingestion.parser import (
    ParsedDocument,
    ParsedImageBlock,
    ParsedTableBlock,
    ParsedTextBlock,
)


def _document(*, tables=None, images=None) -> ParsedDocument:
    return ParsedDocument(
        doc_title="Atlas Router",
        doc_type="product_info",
        source_path=Path("atlas.pdf"),
        blocks=[ParsedTextBlock(text="Overview", page_number=1)],
        tables=tables or [],
        images=images or [],
    )


def test_tables_are_graph_eligible_captions_are_not() -> None:
    document = _document(
        tables=[ParsedTableBlock(markdown="| a | b |\n| - | - |\n| 1 | 2 |", page_number=2)],
        images=[ParsedImageBlock(caption="A diagram of the router.", page_number=3)],
    )

    result = MultimodalIngestor().build_chunks(document)

    assert [chunk.content_type for chunk in result.graph_eligible] == ["table"]
    assert [chunk.content_type for chunk in result.caption_only] == ["image_caption"]
    assert result.table_count == 1
    assert result.image_count == 1


def test_chart_derived_table_is_graph_eligible() -> None:
    document = _document(
        images=[
            ParsedImageBlock(
                caption="Revenue chart.",
                page_number=4,
                chart_markdown="| q | rev |\n| - | - |\n| Q1 | 100 |",
            )
        ]
    )

    result = MultimodalIngestor().build_chunks(document)

    # The chart's tabular data joins the graph-eligible set; the caption does not.
    assert [chunk.content_type for chunk in result.graph_eligible] == ["table"]
    assert [chunk.content_type for chunk in result.caption_only] == ["image_caption"]
    # image_count counts captioned pictures, not the derived table chunk.
    assert result.image_count == 1


def test_all_preserves_table_then_caption_order() -> None:
    document = _document(
        tables=[ParsedTableBlock(markdown="| a |\n| - |\n| 1 |", page_number=1)],
        images=[ParsedImageBlock(caption="A figure.", page_number=2)],
    )

    result = MultimodalIngestor().build_chunks(document)

    assert [chunk.content_type for chunk in result.all] == ["table", "image_caption"]
