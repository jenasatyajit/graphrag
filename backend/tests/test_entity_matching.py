"""Regression tests for the tightened graph entity matcher (VERIFICATION §9.1).

Covers the two citation-quality bugs the loose bidirectional-``CONTAINS`` match
caused: the "Employee" vs "Employee Onboarding" wrong-node match, and spec-label
/SKU nodes (ATL-5000, Target, Throughput, ...) leaking in as matched entities.
A positive guard confirms legitimate fuzzy matches still resolve.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.api.config import Settings
from src.indexing.graph_store import GraphStore, _best_entity_match
from tests.conftest import settings_values
from tests.test_graph_store import FakeDriver


def _settings() -> Settings:
    return Settings(_env_file=None, **settings_values())


# --- pure helper (no DB) ---------------------------------------------------

_ONBOARDING_NODES = ["Employee", "Employee Onboarding", "Expense Approval SOP"]


def test_multi_word_candidate_prefers_most_specific_node() -> None:
    # "Employee Onboarding" must win over the shorter "Employee" node.
    assert _best_entity_match("Employee Onboarding", _ONBOARDING_NODES) == "Employee Onboarding"


def test_single_word_candidate_does_not_expand_to_multi_word_node() -> None:
    # A bare "Employee" must resolve to the "Employee" node only, never leak into
    # "Employee Onboarding" or the unrelated "Expense Approval SOP".
    assert _best_entity_match("Employee", _ONBOARDING_NODES) == "Employee"


def test_single_word_candidate_matches_nothing_when_exact_node_absent() -> None:
    # Without an "Employee" node present, a bare "Employee" must not fall through
    # to the multi-word specialization (the §9.1 wrong-match).
    assert _best_entity_match("Employee", ["Employee Onboarding", "Expense Approval SOP"]) is None


_ATLAS_NODES = [
    "Atlas 5000 Router",
    "ATL-5000",
    "Target",
    "Simultaneous Tunnels",
    "Throughput",
    "Port Count",
    "Dual-WAN Failover",
]


@pytest.mark.parametrize("sku_or_spec", ["ATL-5000", "Target", "Throughput", "Port Count"])
def test_spec_label_and_sku_nodes_do_not_leak(sku_or_spec: str) -> None:
    # The Atlas product candidate must never resolve to a SKU or spec-label node.
    assert _best_entity_match("Atlas 5000 Router", _ATLAS_NODES) != sku_or_spec


def test_atlas_candidate_resolves_to_the_real_product_node() -> None:
    assert _best_entity_match("Atlas 5000 Router", _ATLAS_NODES) == "Atlas 5000 Router"


def test_feature_candidate_resolves_to_the_feature_node() -> None:
    assert _best_entity_match("Dual-WAN Failover", _ATLAS_NODES) == "Dual-WAN Failover"


# --- positive guards: legitimate fuzzy matches survive ---------------------


def test_finance_approval_still_matches_the_step_node() -> None:
    nodes = ["Finance Approval", "Manager Review", "Expense Approval SOP"]
    assert _best_entity_match("Finance Approval", nodes) == "Finance Approval"


def test_single_word_node_matches_multi_word_candidate() -> None:
    # "SAP system" -> "SAP" (single-word node ⊆ multi-word candidate) stays allowed.
    assert _best_entity_match("SAP system", ["SAP", "Concur"]) == "SAP"


# --- end-to-end through match_entities with the FakeDriver -----------------


@pytest.mark.asyncio
async def test_match_entities_filters_spec_leak_end_to_end() -> None:
    # The Cypher pre-filter (containment) would surface the SKU/spec nodes; the
    # Python matcher must drop them and keep only the real product node.
    candidate_rows: list[dict[str, Any]] = [{"name": name} for name in _ATLAS_NODES]
    driver = FakeDriver(results=[candidate_rows])
    store = GraphStore(_settings(), driver=driver)

    matched = await store.match_entities(["Atlas 5000 Router"])

    assert matched == ["Atlas 5000 Router"]
    assert "ATL-5000" not in matched
    assert "Throughput" not in matched
    # Read routing + parameterized query (no string-concatenated values).
    assert driver.calls[0][2] == "r"
    assert driver.calls[0][1]["names"] == ["Atlas 5000 Router"]


@pytest.mark.asyncio
async def test_match_entities_onboarding_disambiguation_end_to_end() -> None:
    candidate_rows: list[dict[str, Any]] = [{"name": name} for name in _ONBOARDING_NODES]
    driver = FakeDriver(results=[candidate_rows])
    store = GraphStore(_settings(), driver=driver)

    matched = await store.match_entities(["Employee Onboarding"])

    assert matched == ["Employee Onboarding"]
