"""Phase 3 graph search, fusion, and SOP version-warning behavior."""

from __future__ import annotations

import json
from collections.abc import Sequence
from uuid import UUID

import pytest

from src.api.schemas import Citation, CitationReference
from src.generation.provenance import (
    CitableSource,
    build_version_warning,
    ground_internal_answer,
)
from src.generation.synthesizer import AnswerSynthesizer, CompletionResult
from src.indexing.graph_schema import GraphFact
from src.indexing.vector_store import RetrievedChunk
from src.ingestion.chunker import Chunk
from src.retrieval.graph_search import GraphSearchService


class ScriptedCompletionProvider:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.prompts: list[str] = []

    async def complete(self, *, system_prompt: str, user_prompt: str) -> CompletionResult:
        del system_prompt
        self.prompts.append(user_prompt)
        return CompletionResult(text=self._reply)


class StubGraphStore:
    def __init__(self, matched: list[str], facts: list[GraphFact]) -> None:
        self._matched = matched
        self._facts = facts
        self.match_calls: list[list[str]] = []

    async def match_entities(self, names: Sequence[str]) -> list[str]:
        self.match_calls.append(list(names))
        return self._matched

    async def neighborhood_facts(self, node_names: Sequence[str]) -> list[GraphFact]:
        del node_names
        return self._facts


class ExplodingGraphStore:
    async def match_entities(self, names: Sequence[str]) -> list[str]:
        raise RuntimeError("neo4j down")

    async def neighborhood_facts(self, node_names: Sequence[str]) -> list[GraphFact]:
        raise RuntimeError("neo4j down")


def _fact() -> GraphFact:
    return GraphFact(
        subject_name="Finance Approval",
        relation="REQUIRES",
        object_name="SAP",
        chunk_ids=["chunk-1"],
    )


@pytest.mark.asyncio
async def test_graph_search_extracts_matches_and_traverses() -> None:
    provider = ScriptedCompletionProvider(json.dumps({"entities": ["SAP", "Finance Approval"]}))
    store = StubGraphStore(matched=["SAP"], facts=[_fact()])
    service = GraphSearchService(store, provider)  # type: ignore[arg-type]

    facts = await service.search("Which system does finance approval require?")

    assert len(facts) == 1
    assert facts[0].as_text() == "Finance Approval requires SAP"
    assert store.match_calls == [["SAP", "Finance Approval"]]


@pytest.mark.asyncio
async def test_graph_search_no_entities_returns_empty() -> None:
    provider = ScriptedCompletionProvider(json.dumps({"entities": []}))
    store = StubGraphStore(matched=[], facts=[])
    service = GraphSearchService(store, provider)  # type: ignore[arg-type]

    facts = await service.search("hello there")

    assert facts == []


@pytest.mark.asyncio
async def test_graph_search_degrades_gracefully_on_failure() -> None:
    provider = ScriptedCompletionProvider(json.dumps({"entities": ["SAP"]}))
    service = GraphSearchService(ExplodingGraphStore(), provider)  # type: ignore[arg-type]

    facts = await service.search("Which system does finance approval require?")

    # A graph outage must not raise; it degrades to no graph facts.
    assert facts == []


@pytest.mark.asyncio
async def test_graph_search_handles_non_json_entity_reply() -> None:
    provider = ScriptedCompletionProvider("I think the entities are SAP and Concur.")
    store = StubGraphStore(matched=["SAP"], facts=[_fact()])
    service = GraphSearchService(store, provider)  # type: ignore[arg-type]

    facts = await service.search("anything")

    # Unparseable extractor output degrades to no entities, hence no facts.
    assert facts == []


# --- fusion + citation grounding ------------------------------------------


def _chunk(
    marker: str, text: str, *, is_current: bool = True, version: str | None = "v2.1"
) -> Chunk:
    return Chunk(
        chunk_id=UUID(f"00000000-0000-0000-0000-0000000000{marker}"),
        doc_id=UUID("00000000-0000-0000-0000-0000000000ff"),
        doc_title="Expense Approval SOP",
        doc_type="SOP",
        text=text,
        sop_version=version,
        is_current_version=is_current,
    )


def _retrieved(chunk: Chunk) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, score=0.9)


@pytest.mark.asyncio
async def test_synthesizer_fuses_graph_facts_as_citable_sources() -> None:
    # Answer cites [1] (chunk) and [2] (graph fact).
    provider = ScriptedCompletionProvider(
        "Managers approve expenses [1]. Finance approval requires SAP [2]."
    )
    synthesizer = AnswerSynthesizer(provider)  # type: ignore[arg-type]
    chunk = _chunk("01", "Managers approve expenses.")
    sources = [_retrieved(chunk)]
    # The graph fact is extracted from the retrieved chunk, so it resolves to the
    # real source document rather than being dropped.
    fact = GraphFact(
        subject_name="Finance Approval",
        relation="REQUIRES",
        object_name="SAP",
        chunk_ids=[str(chunk.chunk_id)],
    )

    grounded = await synthesizer.synthesize(
        "How are expenses approved?",
        sources,
        graph_facts=[fact],
    )

    assert len(grounded.citations) == 2
    # The graph-fact citation resolves to the source chunk id (chunk-based).
    graph_citation = grounded.citations[1]
    assert graph_citation.reference.chunk_id == str(chunk.chunk_id)


def test_unresolvable_graph_fact_is_dropped() -> None:
    # A graph fact with no chunk id (or whose chunk is not in the retrieved set)
    # cannot resolve a real source document, so it is dropped rather than cited
    # with a misleading generic "Knowledge graph" label.
    chunkless = CitableSource.from_graph_fact(
        GraphFact(subject_name="A", relation="HAS_STEP", object_name="B", chunk_ids=[]),
        chunk_lookup={},
    )
    assert chunkless is None

    unretrieved = CitableSource.from_graph_fact(
        GraphFact(subject_name="A", relation="HAS_STEP", object_name="B", chunk_ids=["missing"]),
        chunk_lookup={"other": _retrieved(_chunk("09", "Unrelated text."))},
    )
    assert unretrieved is None

    # Grounding over an empty source list produces no citation and no crash.
    grounded = ground_internal_answer("A has step B [1].", [])
    assert grounded.citations == []


def test_uncited_claim_is_dropped_even_with_graph_context() -> None:
    # This is the SPEC 6.3 hard rule: an uncited sentence never survives, and a
    # graph fact in context does not count as a citation for an unmarked claim.
    fact_source = CitableSource.from_graph_fact(
        _fact(),
        chunk_lookup={"chunk-1": _retrieved(_chunk("01", "Finance approval requires SAP."))},
    )
    assert fact_source is not None
    grounded = ground_internal_answer(
        "Finance approval requires SAP.",  # no [n] marker
        [fact_source],
    )
    assert grounded.citations == []
    assert grounded.flagged_claims  # the unmarked claim is flagged, not surfaced


def test_version_warning_fires_for_superseded_source() -> None:
    outdated_source = CitableSource.from_chunk(
        _retrieved(_chunk("02", "Old policy text.", is_current=False, version="v2.1"))
    )
    grounded = ground_internal_answer("Managers approve expenses [1].", [outdated_source])

    assert grounded.version_warning is not None
    assert "superseded" in grounded.version_warning.lower()


def test_no_version_warning_for_current_source() -> None:
    current = CitableSource.from_chunk(
        _retrieved(_chunk("03", "Current text.", is_current=True, version="v3.0"))
    )
    grounded = ground_internal_answer("Managers approve expenses [1].", [current])

    assert grounded.version_warning is None


def test_build_version_warning_helper() -> None:
    citations = [
        Citation(
            claim_text="x",
            source_tier="internal",
            reference=CitationReference(
                doc_title="Expense Approval SOP",
                is_current_version=False,
            ),
        )
    ]
    assert build_version_warning(citations) is not None
    assert build_version_warning([]) is None
