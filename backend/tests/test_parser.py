"""Docling parser boundary tests using a fake converter."""

from types import SimpleNamespace

import pytest

from src.api.errors import IngestionError
from src.ingestion.parser import DocumentParser


def test_parser_preserves_narrative_text_page_provenance(tmp_path) -> None:
    source = tmp_path / "returns.pdf"
    source.write_bytes(b"fake")
    items = [
        SimpleNamespace(
            text="Step 1: Inspect the item",
            label=SimpleNamespace(value="section_header"),
            prov=[SimpleNamespace(page_no=4)],
        ),
        SimpleNamespace(text=" ", label=None, prov=[]),
    ]
    converter = SimpleNamespace(
        convert=lambda path: SimpleNamespace(document=SimpleNamespace(texts=items))
    )

    document = DocumentParser(converter_factory=lambda: converter).parse(source, doc_type="SOP")

    assert document.doc_title == "returns"
    assert len(document.blocks) == 1
    assert document.blocks[0].page_number == 4
    assert document.blocks[0].label == "section_header"


def test_parser_accepts_markdown_without_page_numbers(tmp_path) -> None:
    source = tmp_path / "policy.md"
    source.write_text("# Policy\n\nRefunds require approval.", encoding="utf-8")
    items = [
        SimpleNamespace(
            text="Refunds require approval.",
            label=SimpleNamespace(value="text"),
            prov=[],
        ),
    ]
    converter = SimpleNamespace(
        convert=lambda path: SimpleNamespace(document=SimpleNamespace(texts=items))
    )

    document = DocumentParser(converter_factory=lambda: converter).parse(
        source, doc_type="product_info"
    )

    assert document.doc_title == "policy"
    assert len(document.blocks) == 1
    assert document.blocks[0].page_number is None


def test_parser_rejects_unsupported_files(tmp_path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("text", encoding="utf-8")

    with pytest.raises(IngestionError, match="format is not supported"):
        DocumentParser().parse(source, doc_type="SOP")
