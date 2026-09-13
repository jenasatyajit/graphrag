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


def test_parser_rejects_unsupported_files(tmp_path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("text", encoding="utf-8")

    with pytest.raises(IngestionError, match="format is not supported"):
        DocumentParser().parse(source, doc_type="SOP")
