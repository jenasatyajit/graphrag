"""Docling parser boundary tests using a fake converter."""

from types import SimpleNamespace

import pytest

from src.api.errors import IngestionError
from src.ingestion.parser import DocumentParser, _title_from_stem


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


def test_parser_uses_explicit_doc_title_verbatim(tmp_path) -> None:
    # A version-suffixed filename would otherwise collapse to a stem-derived
    # title; an explicit doc_title must win untouched.
    source = tmp_path / "expense_approval_sop_v3.md"
    source.write_text("# SOP\n\nApprove expenses.", encoding="utf-8")
    items = [SimpleNamespace(text="Approve expenses.", label=None, prov=[])]
    converter = SimpleNamespace(
        convert=lambda path: SimpleNamespace(document=SimpleNamespace(texts=items))
    )

    document = DocumentParser(converter_factory=lambda: converter).parse(
        source, doc_type="SOP", doc_title="Expense Approval SOP"
    )

    assert document.doc_title == "Expense Approval SOP"


def test_parser_strips_version_suffix_from_stem_fallback(tmp_path) -> None:
    # Without an explicit doc_title, sibling version files must collapse to the
    # same title so the SUPERSEDES chain can match them.
    source = tmp_path / "expense_approval_sop_v3.md"
    source.write_text("# SOP\n\nApprove expenses.", encoding="utf-8")
    items = [SimpleNamespace(text="Approve expenses.", label=None, prov=[])]
    converter = SimpleNamespace(
        convert=lambda path: SimpleNamespace(document=SimpleNamespace(texts=items))
    )

    document = DocumentParser(converter_factory=lambda: converter).parse(source, doc_type="SOP")

    assert document.doc_title == "expense_approval_sop"


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("expense_approval_sop_v2", "expense_approval_sop"),
        ("expense_approval_sop_v3", "expense_approval_sop"),
        ("report_v2.1", "report"),
        ("manual-V3", "manual"),
        ("guide v3.0", "guide"),
        # Non-version stems must be left exactly as-is.
        ("returns", "returns"),
        ("policy", "policy"),
        ("atlas_5000", "atlas_5000"),
        ("model_v", "model_v"),
    ],
)
def test_title_from_stem_strips_only_version_suffix(stem: str, expected: str) -> None:
    assert _title_from_stem(stem) == expected


def test_parser_repairs_mojibake(tmp_path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("placeholder", encoding="utf-8")
    # Build genuine mojibake: real UTF-8 bytes of an en-dash decoded as Latin-1.
    mojibake = "value \u2013 price".encode("utf-8").decode("latin-1")
    assert "\u00e2\u0080" in mojibake
    items = [SimpleNamespace(text=mojibake, label=None, prov=[])]
    converter = SimpleNamespace(
        convert=lambda path: SimpleNamespace(document=SimpleNamespace(texts=items))
    )

    document = DocumentParser(converter_factory=lambda: converter).parse(
        source, doc_type="product_info"
    )

    assert document.blocks[0].text == "value - price"


def test_parser_normalizes_typographic_punctuation(tmp_path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("placeholder", encoding="utf-8")
    items = [
        SimpleNamespace(
            text="it\u2019s a \u201cquoted\u201d value\u2026",
            label=None,
            prov=[],
        ),
    ]
    converter = SimpleNamespace(
        convert=lambda path: SimpleNamespace(document=SimpleNamespace(texts=items))
    )

    document = DocumentParser(converter_factory=lambda: converter).parse(
        source, doc_type="product_info"
    )

    assert document.blocks[0].text == 'it\'s a "quoted" value...'


def test_parser_rejects_unsupported_files(tmp_path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("text", encoding="utf-8")

    with pytest.raises(IngestionError, match="format is not supported"):
        DocumentParser().parse(source, doc_type="SOP")


def _fake_document(*, texts, tables=None, pictures=None):
    return SimpleNamespace(texts=texts, tables=tables or [], pictures=pictures or [])


def _fake_image(saved_paths):
    """A PIL-like image whose save() records the target path."""

    def save(target, fmt):
        del fmt
        saved_paths.append(target)

    return SimpleNamespace(save=save)


def test_parser_extracts_table_markdown(tmp_path) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"fake")
    table = SimpleNamespace(
        export_to_markdown=lambda doc: "| a | b |\n| - | - |\n| 1 | 2 |",
        prov=[SimpleNamespace(page_no=3)],
    )
    document = _fake_document(
        texts=[SimpleNamespace(text="Overview", label=None, prov=[])],
        tables=[table],
    )
    converter = SimpleNamespace(convert=lambda path: SimpleNamespace(document=document))

    parsed = DocumentParser(converter_factory=lambda: converter).parse(
        source, doc_type="product_info"
    )

    assert len(parsed.tables) == 1
    assert "| a | b |" in parsed.tables[0].markdown
    assert parsed.tables[0].page_number == 3


def test_parser_captions_pictures_and_saves_images(tmp_path) -> None:
    source = tmp_path / "brochure.pdf"
    source.write_bytes(b"fake")
    saved: list = []
    picture = SimpleNamespace(
        caption_text=lambda doc: "A bar chart of quarterly revenue.",
        get_image=lambda doc: _fake_image(saved),
        annotations=[],
        prov=[SimpleNamespace(page_no=5)],
    )
    document = _fake_document(
        texts=[SimpleNamespace(text="Overview", label=None, prov=[])],
        pictures=[picture],
    )
    converter = SimpleNamespace(convert=lambda path: SimpleNamespace(document=document))

    parsed = DocumentParser(
        converter_factory=lambda: converter,
        image_output_dir=tmp_path / "images",
    ).parse(source, doc_type="product_info")

    assert len(parsed.images) == 1
    assert parsed.images[0].caption == "A bar chart of quarterly revenue."
    assert parsed.images[0].page_number == 5
    assert parsed.images[0].image_path is not None
    assert len(saved) == 1


def test_parser_skips_pictures_without_caption(tmp_path) -> None:
    source = tmp_path / "brochure.pdf"
    source.write_bytes(b"fake")
    picture = SimpleNamespace(
        caption_text=lambda doc: "",
        get_image=lambda doc: None,
        annotations=[],
        prov=[SimpleNamespace(page_no=1)],
    )
    document = _fake_document(
        texts=[SimpleNamespace(text="Overview", label=None, prov=[])],
        pictures=[picture],
    )
    converter = SimpleNamespace(convert=lambda path: SimpleNamespace(document=document))

    parsed = DocumentParser(converter_factory=lambda: converter).parse(
        source, doc_type="product_info"
    )

    assert parsed.images == []
