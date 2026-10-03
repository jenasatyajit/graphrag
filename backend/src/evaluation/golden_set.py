"""Curated golden-set model and validating loader (Phase 6, SPEC 11).

The golden set is the ground truth the RAGAS metrics and the ablation study are
scored against. It is a curated, committed artifact (unlike generated eval
reports), authored from the real ``data/samples/`` corpus. The loader validates
every item's structure AND asserts the SPEC 11 required-category minimum counts,
so a set that silently drops a required coverage category fails loudly rather
than producing a misleadingly green run.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

GoldenCategory = Literal["graph", "table_chart", "web_search", "superseded_sop", "general"]
ExpectedTool = Literal["internal_search", "web_search", "answer_directly"]

# Total-count bounds and per-category minimums enforced by the loader (SPEC 11).
# Exposed as a module-level constant so the loader and its tests share one source
# of truth instead of duplicating the numbers.
MIN_TOTAL_ITEMS = 30
MAX_TOTAL_ITEMS = 50
REQUIRED_MINIMUMS: dict[GoldenCategory, int] = {
    "graph": 5,
    "table_chart": 5,
    "web_search": 2,
    "superseded_sop": 1,
}


class GoldenItem(BaseModel):
    """One evaluation case: a question with its ground-truth answer and contexts."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    ground_truth_answer: str = Field(min_length=1)
    ground_truth_contexts: list[str] = Field(min_length=1)
    category: GoldenCategory
    expected_tool: ExpectedTool | None = None
    # True for web-search items whose ground-truth contexts are web-derived and
    # therefore NOT answerable from the internal corpus (external by design).
    contexts_are_external: bool = False
    notes: str | None = None


class GoldenSet(BaseModel):
    """A validated collection of golden items."""

    items: list[GoldenItem]

    def category_counts(self) -> dict[str, int]:
        """Return the number of items per category."""
        return {
            str(category): count
            for category, count in Counter(item.category for item in self.items).items()
        }


def load_golden_set(path: Path) -> GoldenSet:
    """Load, structurally validate, and coverage-check a golden set.

    Accepts either JSONL (one ``GoldenItem`` object per non-blank line) or a JSON
    array of items (chosen by the file suffix). Raises a clear ``ValueError`` when
    the file is missing, malformed, outside the 30-50 total bound, or short on any
    SPEC 11 required category (the message names the deficient category).
    """
    if not path.exists():
        raise ValueError(f"Golden set not found at {path}.")

    raw_items = _read_items(path)
    items = [GoldenItem.model_validate(entry) for entry in raw_items]
    golden = GoldenSet(items=items)
    _assert_coverage(golden)
    return golden


def _read_items(path: Path) -> list[object]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("A .json golden set must contain a top-level array of items.")
        return parsed
    items: list[object] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            items.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return items


def _assert_coverage(golden: GoldenSet) -> None:
    total = len(golden.items)
    if total < MIN_TOTAL_ITEMS or total > MAX_TOTAL_ITEMS:
        raise ValueError(
            f"Golden set must contain {MIN_TOTAL_ITEMS}-{MAX_TOTAL_ITEMS} items; found {total}."
        )
    counts = golden.category_counts()
    for category, minimum in REQUIRED_MINIMUMS.items():
        found = counts.get(category, 0)
        if found < minimum:
            raise ValueError(
                f"Golden set requires at least {minimum} '{category}' items; found {found}."
            )
    ids = [item.id for item in golden.items]
    duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicates:
        raise ValueError(f"Golden set contains duplicate item ids: {', '.join(duplicates)}.")
