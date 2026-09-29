"""Phase 3 Neo4j graph store: writes, versioning, matching, and traversal."""

from __future__ import annotations

from typing import Any

import pytest

from src.api.config import Settings
from src.api.errors import GraphError
from src.indexing.graph_schema import GraphNode, GraphTriple
from src.indexing.graph_store import GraphStore
from tests.conftest import settings_values


class FakeRecord:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def data(self) -> dict[str, Any]:
        return self._data


class FakeResult:
    def __init__(self, records: list[FakeRecord]) -> None:
        self.records = records


class FakeDriver:
    """Records executed queries and replays scripted results in order."""

    def __init__(self, results: list[list[dict[str, Any]]] | None = None) -> None:
        self._results = list(results or [])
        self.calls: list[tuple[str, dict[str, Any], str]] = []
        self.closed = False
        self.connectivity_error: Exception | None = None

    async def verify_connectivity(self) -> None:
        if self.connectivity_error is not None:
            raise self.connectivity_error

    async def execute_query(
        self,
        cypher: str,
        *,
        parameters_: dict[str, Any],
        database_: str,
        routing_: str,
    ) -> FakeResult:
        del database_
        self.calls.append((cypher, parameters_, routing_))
        records = self._results.pop(0) if self._results else []
        return FakeResult([FakeRecord(row) for row in records])

    async def close(self) -> None:
        self.closed = True


def _settings() -> Settings:
    return Settings(_env_file=None, **settings_values())


def _triple(sub_label: str, sub: str, rel: str, obj_label: str, obj: str) -> GraphTriple:
    return GraphTriple(
        subject=GraphNode(label=sub_label, name=sub),  # type: ignore[arg-type]
        relation=rel,  # type: ignore[arg-type]
        object=GraphNode(label=obj_label, name=obj),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_verify_connectivity_ok() -> None:
    store = GraphStore(_settings(), driver=FakeDriver())
    health = await store.verify_connectivity()
    assert health.status == "ok"
    assert health.name == "neo4j"


@pytest.mark.asyncio
async def test_verify_connectivity_unhealthy() -> None:
    driver = FakeDriver()
    driver.connectivity_error = RuntimeError("down")
    store = GraphStore(_settings(), driver=driver)
    health = await store.verify_connectivity()
    assert health.status == "unhealthy"


@pytest.mark.asyncio
async def test_write_triples_groups_by_shape_and_uses_write_routing() -> None:
    driver = FakeDriver(results=[[{"edges": 1}], [{"edges": 1}]])
    store = GraphStore(_settings(), driver=driver)

    triples = [
        _triple("Process", "Expense Approval", "HAS_STEP", "Step", "Manager Review"),
        _triple("Step", "Manager Review", "OWNED_BY", "Role", "Manager"),
    ]
    written = await store.write_triples(triples, chunk_id="chunk-1")

    assert written == 2
    # Two distinct edge shapes -> two grouped write queries.
    assert len(driver.calls) == 2
    for cypher, params, routing in driver.calls:
        assert routing == "w"
        assert params["chunk_id"] == "chunk-1"
        # Labels/relations are interpolated as schema constants only.
        assert "row.subject_name" in cypher


@pytest.mark.asyncio
async def test_process_merge_is_version_aware() -> None:
    # A Process carrying sop_version must MERGE on (name, sop_version) so two SOP
    # versions become two distinct nodes; other labels MERGE on name only. This
    # is what lets the SUPERSEDES chain link a superseded version to the current.
    driver = FakeDriver(results=[[{"edges": 1}]])
    store = GraphStore(_settings(), driver=driver)

    triple = GraphTriple(
        subject=GraphNode(
            label="Process",
            name="Expense Approval SOP",
            properties={"sop_version": "v3.0", "doc_title": "Expense Approval SOP"},
        ),
        relation="HAS_STEP",
        object=GraphNode(label="Step", name="Manager Review"),
    )
    await store.write_triples([triple], chunk_id="chunk-1")

    cypher, params, _ = driver.calls[0]
    # Process MERGE keys on both name and version; Step MERGE keys on name only.
    assert "sop_version: row.subject_version" in cypher
    assert "MERGE (o:`Step` {name: row.object_name})" in cypher
    assert params["rows"][0]["subject_version"] == "v3.0"
    assert params["rows"][0]["object_version"] == ""


@pytest.mark.asyncio
async def test_supersede_previous_versions_creates_edge() -> None:
    driver = FakeDriver(results=[[{"superseded": 1}]])
    store = GraphStore(_settings(), driver=driver)

    count = await store.supersede_previous_versions("Expense Approval SOP", "v3.0")

    assert count == 1
    cypher, params, routing = driver.calls[0]
    assert routing == "w"
    assert "SUPERSEDES" in cypher
    assert "is_current_version = false" in cypher
    assert params["current_version"] == "v3.0"


@pytest.mark.asyncio
async def test_superseded_chunk_ids() -> None:
    driver = FakeDriver(results=[[{"chunk_ids": ["a", "b"]}]])
    store = GraphStore(_settings(), driver=driver)

    ids = await store.superseded_chunk_ids()

    assert ids == {"a", "b"}
    assert driver.calls[0][2] == "r"


@pytest.mark.asyncio
async def test_match_entities_returns_names() -> None:
    driver = FakeDriver(results=[[{"name": "SAP"}, {"name": "Concur"}]])
    store = GraphStore(_settings(), driver=driver)

    names = await store.match_entities(["sap", "concur system"])

    assert names == ["SAP", "Concur"]
    assert driver.calls[0][2] == "r"


@pytest.mark.asyncio
async def test_neighborhood_facts_filters_off_schema_relations() -> None:
    driver = FakeDriver(
        results=[
            [
                {
                    "subject_name": "Finance Approval",
                    "relation": "REQUIRES",
                    "object_name": "SAP",
                    "chunk_ids": ["chunk-1", "chunk-1", "chunk-2"],
                },
                {
                    "subject_name": "Finance Approval",
                    "relation": "MENTIONS",  # not in schema -> dropped
                    "object_name": "Ledger",
                    "chunk_ids": [],
                },
            ]
        ]
    )
    store = GraphStore(_settings(), driver=driver)

    facts = await store.neighborhood_facts(["Finance Approval"])

    assert len(facts) == 1
    assert facts[0].relation == "REQUIRES"
    assert facts[0].object_name == "SAP"
    # chunk_ids are de-duplicated.
    assert facts[0].chunk_ids == ["chunk-1", "chunk-2"]
    assert facts[0].as_text() == "Finance Approval requires SAP"


@pytest.mark.asyncio
async def test_match_entities_empty_input_skips_query() -> None:
    driver = FakeDriver()
    store = GraphStore(_settings(), driver=driver)

    names = await store.match_entities(["  ", ""])

    assert names == []
    assert driver.calls == []


@pytest.mark.asyncio
async def test_query_failure_wraps_in_graph_error() -> None:
    class ExplodingDriver(FakeDriver):
        async def execute_query(self, *args: Any, **kwargs: Any) -> FakeResult:
            raise RuntimeError("boom")

    store = GraphStore(_settings(), driver=ExplodingDriver())

    with pytest.raises(GraphError):
        await store.match_entities(["SAP"])


@pytest.mark.asyncio
async def test_close_only_closes_owned_driver() -> None:
    injected = FakeDriver()
    store = GraphStore(_settings(), driver=injected)
    await store.close()
    # Injected drivers are not owned by the store, so they stay open.
    assert injected.closed is False
