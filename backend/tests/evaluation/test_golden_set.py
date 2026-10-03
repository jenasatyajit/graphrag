"""Tests for the golden-set model and coverage-enforcing loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.golden_set import (
    REQUIRED_MINIMUMS,
    GoldenItem,
    load_golden_set,
)


def _item(item_id: str, category: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": item_id,
        "question": f"Question {item_id}?",
        "ground_truth_answer": f"Answer {item_id}.",
        "ground_truth_contexts": [f"Context for {item_id}."],
        "category": category,
    }
    payload.update(overrides)
    return payload


def _balanced_items(total: int = 34) -> list[dict[str, object]]:
    """Build a set that meets every required minimum, padded with 'general'."""
    items: list[dict[str, object]] = []
    counter = 0
    for category, minimum in REQUIRED_MINIMUMS.items():
        for _ in range(minimum):
            items.append(_item(f"{category}-{counter}", category))
            counter += 1
    while len(items) < total:
        items.append(_item(f"general-{counter}", "general"))
        counter += 1
    return items


def _write_jsonl(path: Path, items: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(item) for item in items) + "\n",
        encoding="utf-8",
    )
    return path


def test_loads_a_valid_set(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "golden.jsonl", _balanced_items())

    golden = load_golden_set(path)

    assert len(golden.items) == 34
    counts = golden.category_counts()
    for category, minimum in REQUIRED_MINIMUMS.items():
        assert counts[category] >= minimum


def test_loads_a_json_array(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(_balanced_items()), encoding="utf-8")

    golden = load_golden_set(path)

    assert len(golden.items) == 34


@pytest.mark.parametrize("missing", sorted(REQUIRED_MINIMUMS))
def test_rejects_set_missing_a_required_category(tmp_path: Path, missing: str) -> None:
    items = [item for item in _balanced_items() if item["category"] != missing]
    # Keep the total within bounds by padding general items back to 30+.
    while len(items) < 30:
        items.append(_item(f"pad-{len(items)}", "general"))
    path = _write_jsonl(tmp_path / "golden.jsonl", items)

    with pytest.raises(ValueError, match=missing):
        load_golden_set(path)


def test_rejects_too_few_items(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "golden.jsonl", _balanced_items(total=29))

    with pytest.raises(ValueError, match="30-50"):
        load_golden_set(path)


def test_rejects_too_many_items(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "golden.jsonl", _balanced_items(total=51))

    with pytest.raises(ValueError, match="30-50"):
        load_golden_set(path)


def test_rejects_duplicate_ids(tmp_path: Path) -> None:
    items = _balanced_items()
    items[-1]["id"] = items[0]["id"]
    path = _write_jsonl(tmp_path / "golden.jsonl", items)

    with pytest.raises(ValueError, match="duplicate"):
        load_golden_set(path)


def test_missing_file_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        load_golden_set(tmp_path / "nope.jsonl")


def test_item_requires_at_least_one_context() -> None:
    with pytest.raises(ValueError):
        GoldenItem(
            id="x",
            question="q?",
            ground_truth_answer="a",
            ground_truth_contexts=[],
            category="general",
        )


def test_committed_golden_set_loads() -> None:
    """The real committed golden set must satisfy the loader's requirements."""
    golden = load_golden_set(Path("data/eval/golden_set.jsonl"))

    assert 30 <= len(golden.items) <= 50
    counts = golden.category_counts()
    for category, minimum in REQUIRED_MINIMUMS.items():
        assert counts[category] >= minimum
    external = [item for item in golden.items if item.category == "web_search"]
    assert all(item.contexts_are_external for item in external)
