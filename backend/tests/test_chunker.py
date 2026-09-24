"""SOP-aware and fallback chunking behavior tests."""

from pathlib import Path
from uuid import UUID

from src.ingestion.chunker import DocumentChunker
from src.ingestion.parser import ParsedDocument, ParsedTextBlock


def test_sop_chunks_by_step_and_preserves_order() -> None:
    document = ParsedDocument(
        doc_id=UUID("00000000-0000-0000-0000-000000000010"),
        doc_title="Onboarding",
        doc_type="SOP",
        source_path=Path("onboarding.pdf"),
        blocks=[
            ParsedTextBlock(text="Step 1: Create account", page_number=1, label="section_header"),
            ParsedTextBlock(text="The IT admin creates the account.", page_number=1),
            ParsedTextBlock(text="Step 2: Verify access", page_number=2, label="section_header"),
            ParsedTextBlock(text="The manager verifies access.", page_number=2),
        ],
        sop_version="v1.0",
    )

    chunks = DocumentChunker().chunk(document)

    assert [chunk.step_order for chunk in chunks] == [1, 2]
    assert chunks[0].page_number == 1
    assert "IT admin" in chunks[0].text
    assert chunks[1].page_number == 2
    assert all(chunk.source_tier == "internal" for chunk in chunks)


def test_unstructured_prose_uses_bounded_overlapping_fallback() -> None:
    text = " ".join(f"word-{index}" for index in range(80))
    document = ParsedDocument(
        doc_title="General Policy",
        doc_type="SOP",
        source_path=Path("policy.pdf"),
        blocks=[ParsedTextBlock(text=text, page_number=3)],
    )

    chunks = DocumentChunker(max_chars=120, overlap_chars=20).chunk(document)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 120 for chunk in chunks)
    assert all(chunk.step_order is None for chunk in chunks)


def test_long_sop_step_remains_one_chunk() -> None:
    document = ParsedDocument(
        doc_title="Long Procedure",
        doc_type="SOP",
        source_path=Path("long.pdf"),
        blocks=[
            ParsedTextBlock(text="Step 1: Perform checks", page_number=1),
            ParsedTextBlock(text=" ".join(["check"] * 100), page_number=1),
        ],
    )

    chunks = DocumentChunker(max_chars=100, overlap_chars=10).chunk(document)

    assert len(chunks) == 1
    assert chunks[0].step_order == 1


def test_chunk_tables_emits_one_table_chunk_each() -> None:
    from src.ingestion.parser import ParsedTableBlock

    document = ParsedDocument(
        doc_title="Product Specs",
        doc_type="product_info",
        source_path=Path("specs.pdf"),
        blocks=[ParsedTextBlock(text="Overview", page_number=1)],
        tables=[
            ParsedTableBlock(markdown="| a | b |\n| - | - |\n| 1 | 2 |", page_number=2),
            ParsedTableBlock(markdown="| x | y |\n| - | - |\n| 3 | 4 |", page_number=3),
        ],
    )

    chunks = DocumentChunker().chunk_tables(document)

    assert len(chunks) == 2
    assert all(chunk.content_type == "table" for chunk in chunks)
    assert chunks[0].page_number == 2
    assert "| a | b |" in chunks[0].text
    assert all(chunk.source_tier == "internal" for chunk in chunks)


def test_chunk_images_emits_caption_chunk_with_image_reference() -> None:
    from src.ingestion.parser import ParsedImageBlock

    document = ParsedDocument(
        doc_title="Brochure",
        doc_type="product_info",
        source_path=Path("brochure.pdf"),
        blocks=[ParsedTextBlock(text="Overview", page_number=1)],
        images=[
            ParsedImageBlock(
                caption="A bar chart of quarterly revenue.",
                page_number=4,
                image_path="/data/processed/images/doc/picture-1.png",
            )
        ],
    )

    chunks = DocumentChunker().chunk_images(document)

    assert len(chunks) == 1
    assert chunks[0].content_type == "image_caption"
    assert chunks[0].text == "A bar chart of quarterly revenue."
    assert chunks[0].page_number == 4
    assert chunks[0].image_path == "/data/processed/images/doc/picture-1.png"


def test_chunk_images_emits_extra_table_chunk_for_chart_data() -> None:
    from src.ingestion.parser import ParsedImageBlock

    document = ParsedDocument(
        doc_title="Brochure",
        doc_type="product_info",
        source_path=Path("brochure.pdf"),
        blocks=[ParsedTextBlock(text="Overview", page_number=1)],
        images=[
            ParsedImageBlock(
                caption="Quarterly revenue chart.",
                page_number=4,
                image_path=None,
                chart_markdown="| quarter | revenue |\n| - | - |\n| Q1 | 100 |",
            )
        ],
    )

    chunks = DocumentChunker().chunk_images(document)

    content_types = [chunk.content_type for chunk in chunks]
    assert content_types == ["image_caption", "table"]
    assert "| quarter | revenue |" in chunks[1].text
