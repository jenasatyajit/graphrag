"""Argument-wiring tests for the ingestion CLI (no live services)."""

from __future__ import annotations

import importlib

run_ingestion = importlib.import_module("scripts.run_ingestion")


def test_doc_title_flag_is_parsed(monkeypatch) -> None:
    # The --doc-title flag must reach the parsed namespace so main() can thread
    # it through run_ingestion() -> parser.parse(doc_title=...).
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ingestion.py",
            "--doc-type",
            "SOP",
            "--sop-version",
            "v3.0",
            "--doc-title",
            "Expense Approval SOP",
        ],
    )

    args = run_ingestion._parse_args()

    assert args.doc_title == "Expense Approval SOP"


def test_doc_title_defaults_to_none(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_ingestion.py", "--doc-type", "product_info"])

    args = run_ingestion._parse_args()

    assert args.doc_title is None
