"""Phase 7 end-to-end response-shape matrix (SPEC 8), all vendors mocked.

Confirms POST /query returns the exact SPEC 8 shape across the flows the UI
(SPEC 7.8) depends on: internal-only, hybrid+graph, multimodal (table chunk),
web-routed, superseded-SOP, and a multi-turn follow-up. Each flow is driven
through TestClient so the shape asserted is the real serialized HTTP body:
inline citation markers in ``answer``, the internal-vs-external split via each
citation's ``source_tier`` + ``reference`` fields, ``used_web_search``,
``version_warning``, ``tool_trace``, ``conversation_id``, and ``notices``.

Single-flow degradation paths live in test_degradation.py; the always-must-pass
behavior invariants live in test_router.py. This file locks the response
contract so a schema regression is caught before it reaches the frontend.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from src.generation.synthesizer import CompletionResult
from src.indexing.graph_schema import GraphFact
from src.indexing.vector_store import RetrievedChunk
from src.ingestion.chunker import Chunk
from src.retrieval.web_search import WebResult
from tests.conftest import (
    FakeVectorStore,
    FakeWebSearch,
    ScriptedRouterLLM,
    build_client,
    tool_decision,
)

RETRIEVED_AT = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)

# Keys every SPEC 8 response must carry (plus the deviation D-15 conversation
# fields and the notices channel).
_SPEC8_KEYS = {
    "answer",
    "citations",
    "used_web_search",
    "version_warning",
    "tool_trace",
    "conversation_id",
    "notices",
}
_CITATION_KEYS = {"claim_text", "source_tier", "reference"}
_REFERENCE_KEYS = {
    "chunk_id",
    "doc_title",
    "sop_version",
    "is_current_version",
    "url",
    "retrieved_at",
}


def _assert_spec8_shape(payload: dict[str, Any]) -> None:
    assert set(payload) == _SPEC8_KEYS
    assert isinstance(payload["answer"], str) and payload["answer"]
    assert isinstance(payload["used_web_search"], bool)
    assert isinstance(payload["tool_trace"], list)
    assert isinstance(payload["notices"], list)
    assert payload["conversation_id"]
    for citation in payload["citations"]:
        assert set(citation) == _CITATION_KEYS
        assert citation["source_tier"] in {"internal", "external"}
        assert set(citation["reference"]) == _REFERENCE_KEYS


def _web_result() -> WebResult:
    return WebResult(
        title="EU AI Act overview",
        url="https://example.org/ai-act",
        snippet="The EU AI Act entered into force in August 2024.",
        retrieved_at=RETRIEVED_AT,
    )


class PromptAwareCompleter:
    """One provider shared across extraction/synthesis/rewrite/direct answers.

    create_app wires a single completion provider into every LLM-backed slot, so
    the reply is chosen by inspecting the system prompt rather than by call
    order (which would deplete a scripted list across the four slots).
    """

    def __init__(self, *, synth: str, entities: str = '{"entities": []}') -> None:
        self._synth = synth
        self._entities = entities

    async def complete(self, *, system_prompt: str, user_prompt: str) -> CompletionResult:
        del user_prompt
        if "knowledge graph" in system_prompt or '"entities"' in system_prompt:
            return CompletionResult(text=self._entities)
        return CompletionResult(text=self._synth)


# --- internal-only -----------------------------------------------------------


def test_internal_only_flow_returns_spec8_shape() -> None:
    with build_client() as client:
        response = client.post("/query", json={"question": "Who approves large expenses?"})

    assert response.status_code == 200
    payload = response.json()
    _assert_spec8_shape(payload)
    assert "[1]" in payload["answer"]
    assert payload["tool_trace"] == ["internal_search"]
    assert payload["used_web_search"] is False
    assert payload["version_warning"] is None
    citation = payload["citations"][0]
    assert citation["source_tier"] == "internal"
    assert citation["reference"]["chunk_id"] is not None
    assert citation["reference"]["url"] is None


# --- hybrid + graph facts ----------------------------------------------------


class GraphFactStore:
    """Graph store that resolves one entity and returns one citable fact."""

    def __init__(self, chunk_id: str) -> None:
        self._chunk_id = chunk_id

    async def connect(self) -> None:
        return None

    async def verify_connectivity(self) -> Any:
        from src.api.schemas import ComponentHealth

        return ComponentHealth(name="neo4j", status="ok")

    async def match_entities(self, names: Any) -> list[str]:
        del names
        return ["Expense Approval"]

    async def neighborhood_facts(self, node_names: Any) -> list[GraphFact]:
        del node_names
        return [
            GraphFact(
                subject_name="Expense Approval",
                relation="REQUIRES",
                object_name="SAP",
                chunk_ids=[self._chunk_id],
            )
        ]

    async def close(self) -> None:
        return None


def test_hybrid_graph_flow_fuses_graph_fact_as_citation() -> None:
    # The vector hit is [1]; the graph fact is fused as source [2]. Both carry a
    # chunk-based internal citation.
    store = FakeVectorStore()
    chunk_id = str(store.source.chunk.chunk_id)
    completer = PromptAwareCompleter(
        synth="Managers approve expenses [1]. Expense approval requires SAP [2].",
        entities='{"entities": ["expense approval"]}',
    )
    with build_client(
        graph_store=GraphFactStore(chunk_id),
        completion_provider=completer,
    ) as client:
        response = client.post("/query", json={"question": "How is expense approval done?"})

    assert response.status_code == 200
    payload = response.json()
    _assert_spec8_shape(payload)
    assert len(payload["citations"]) == 2
    assert {c["source_tier"] for c in payload["citations"]} == {"internal"}


# --- multimodal (table chunk) ------------------------------------------------


class TableChunkVectorStore(FakeVectorStore):
    """Vector store whose single hit is a Phase 4 table chunk."""

    def __init__(self) -> None:
        super().__init__()
        self.source = RetrievedChunk(
            chunk=Chunk(
                chunk_id=UUID("00000000-0000-0000-0000-0000000000c1"),
                doc_id=UUID("00000000-0000-0000-0000-0000000000c2"),
                doc_title="Atlas Router Datasheet",
                doc_type="product_info",
                text="| Port | Speed |\n| - | - |\n| WAN | 10GbE |",
                page_number=4,
                content_type="table",
            ),
            score=0.88,
            rerank_score=0.77,
        )


def test_multimodal_table_chunk_flow_returns_spec8_shape() -> None:
    completer = PromptAwareCompleter(synth="The WAN port runs at 10GbE [1].")
    with build_client(
        vector_store=TableChunkVectorStore(),
        completion_provider=completer,
    ) as client:
        response = client.post("/query", json={"question": "What speed is the WAN port?"})

    assert response.status_code == 200
    payload = response.json()
    _assert_spec8_shape(payload)
    assert payload["citations"][0]["source_tier"] == "internal"
    assert payload["citations"][0]["reference"]["doc_title"] == "Atlas Router Datasheet"


# --- web-routed --------------------------------------------------------------


def test_web_routed_flow_labels_citation_external() -> None:
    completer = PromptAwareCompleter(
        synth="According to an external source, the Act is in force [1]."
    )
    with build_client(
        completion_provider=completer,
        router_llm=ScriptedRouterLLM(tool_decision("web_search", query="EU AI Act status")),
        web_searcher=FakeWebSearch([_web_result()]),
    ) as client:
        response = client.post("/query", json={"question": "Is the EU AI Act in force?"})

    assert response.status_code == 200
    payload = response.json()
    _assert_spec8_shape(payload)
    assert payload["used_web_search"] is True
    assert payload["tool_trace"] == ["web_search"]
    citation = payload["citations"][0]
    assert citation["source_tier"] == "external"
    assert citation["reference"]["url"] == "https://example.org/ai-act"
    assert citation["reference"]["retrieved_at"] is not None
    assert citation["reference"]["chunk_id"] is None


# --- superseded SOP ----------------------------------------------------------


class SupersededVectorStore(FakeVectorStore):
    """Vector store whose hit is a superseded (is_current_version=false) SOP."""

    def __init__(self) -> None:
        super().__init__()
        self.source = RetrievedChunk(
            chunk=Chunk(
                chunk_id=UUID("00000000-0000-0000-0000-0000000000d1"),
                doc_id=UUID("00000000-0000-0000-0000-0000000000d2"),
                doc_title="Expense Approval SOP",
                doc_type="SOP",
                text="Manager review above $250.",
                page_number=2,
                sop_version="v2.1",
                is_current_version=False,
            ),
            score=0.9,
            rerank_score=0.8,
        )


def test_superseded_sop_flow_surfaces_version_warning() -> None:
    completer = PromptAwareCompleter(synth="Manager review applied above $250 [1].")
    with build_client(
        vector_store=SupersededVectorStore(),
        completion_provider=completer,
    ) as client:
        response = client.post(
            "/query", json={"question": "What was the v2.1 manager review threshold?"}
        )

    assert response.status_code == 200
    payload = response.json()
    _assert_spec8_shape(payload)
    assert payload["version_warning"] is not None
    assert "superseded" in payload["version_warning"].lower()
    assert payload["citations"][0]["reference"]["is_current_version"] is False
    assert payload["citations"][0]["reference"]["sop_version"] == "v2.1"


# --- multi-turn follow-up ----------------------------------------------------


def test_multi_turn_follow_up_keeps_conversation_id_and_spec8_shape() -> None:
    completer = PromptAwareCompleter(synth="Managers approve expenses above $500 [1].")
    with build_client(
        completion_provider=completer,
        router_llm=ScriptedRouterLLM(
            tool_decision("internal_search"),
            tool_decision("internal_search"),
        ),
    ) as client:
        first = client.post("/query", json={"question": "Who approves expenses?"})
        first_payload = first.json()
        _assert_spec8_shape(first_payload)

        conv_id = first_payload["conversation_id"]
        second = client.post(
            "/query",
            json={"question": "and above $1000?", "conversation_id": conv_id},
        )

    assert second.status_code == 200
    second_payload = second.json()
    _assert_spec8_shape(second_payload)
    # The follow-up continues the same conversation thread.
    assert second_payload["conversation_id"] == conv_id
