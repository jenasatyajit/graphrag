"""Phase 3 graph extraction: schema mapping and off-schema rejection."""

from __future__ import annotations

import json
from uuid import UUID

import pytest

from src.api.config import Settings
from src.api.errors import LLMError
from src.generation.synthesizer import CompletionResult
from src.indexing.graph_schema import GraphTriple, is_allowed_edge
from src.ingestion.chunker import Chunk
from src.ingestion.graph_extractor import GraphExtractor
from tests.conftest import settings_values


class ScriptedCompletionProvider:
    """Returns a fixed model reply regardless of prompt."""

    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def complete(self, *, system_prompt: str, user_prompt: str) -> CompletionResult:
        del system_prompt, user_prompt
        return CompletionResult(text=self._reply)


def _settings() -> Settings:
    return Settings(_env_file=None, **settings_values())


def _chunk(text: str, *, sop_version: str | None = "v2.1") -> Chunk:
    return Chunk(
        chunk_id=UUID("00000000-0000-0000-0000-0000000000aa"),
        doc_id=UUID("00000000-0000-0000-0000-0000000000bb"),
        doc_title="Expense Approval SOP",
        doc_type="SOP",
        text=text,
        sop_version=sop_version,
    )


@pytest.mark.asyncio
async def test_extractor_maps_valid_triples() -> None:
    reply = json.dumps(
        {
            "triples": [
                {
                    "subject": {"label": "Process", "name": "Expense Approval", "properties": {}},
                    "relation": "HAS_STEP",
                    "object": {"label": "Step", "name": "Manager Review", "properties": {}},
                },
                {
                    "subject": {"label": "Step", "name": "Manager Review", "properties": {}},
                    "relation": "OWNED_BY",
                    "object": {"label": "Role", "name": "Manager", "properties": {}},
                },
            ]
        }
    )
    extractor = GraphExtractor(_settings(), completion_provider=ScriptedCompletionProvider(reply))

    triples = await extractor.extract(_chunk("Managers review expense reports."))

    assert len(triples) == 2
    assert all(isinstance(triple, GraphTriple) for triple in triples)
    # Process name is canonicalized to the chunk's doc_title.
    assert triples[0].subject.name == "Expense Approval SOP"
    assert triples[0].relation == "HAS_STEP"


@pytest.mark.asyncio
async def test_extractor_enriches_process_nodes_for_versioning() -> None:
    reply = json.dumps(
        {
            "triples": [
                {
                    "subject": {"label": "Process", "name": "Expense Approval", "properties": {}},
                    "relation": "HAS_STEP",
                    "object": {"label": "Step", "name": "Submit Report", "properties": {}},
                }
            ]
        }
    )
    extractor = GraphExtractor(_settings(), completion_provider=ScriptedCompletionProvider(reply))

    triples = await extractor.extract(_chunk("Submit the report.", sop_version="v2.1"))

    process = triples[0].subject
    # Process name is canonicalized to the doc title to prevent fragmentation.
    assert process.name == "Expense Approval SOP"
    assert process.properties["doc_title"] == "Expense Approval SOP"
    assert process.properties["sop_version"] == "v2.1"
    assert process.properties["is_current_version"] is True


@pytest.mark.asyncio
async def test_extractor_drops_off_schema_triples() -> None:
    reply = json.dumps(
        {
            "triples": [
                # Invalid edge shape: Role cannot HAS_STEP a System.
                {
                    "subject": {"label": "Role", "name": "Manager", "properties": {}},
                    "relation": "HAS_STEP",
                    "object": {"label": "System", "name": "SAP", "properties": {}},
                },
                # Invalid label entirely.
                {
                    "subject": {"label": "Widget", "name": "Foo", "properties": {}},
                    "relation": "OWNED_BY",
                    "object": {"label": "Role", "name": "Manager", "properties": {}},
                },
                # One valid triple survives.
                {
                    "subject": {"label": "Step", "name": "Finance Approval", "properties": {}},
                    "relation": "REQUIRES",
                    "object": {"label": "System", "name": "SAP", "properties": {}},
                },
            ]
        }
    )
    extractor = GraphExtractor(_settings(), completion_provider=ScriptedCompletionProvider(reply))

    triples = await extractor.extract(_chunk("Finance approval requires SAP."))

    assert len(triples) == 1
    assert triples[0].relation == "REQUIRES"
    assert triples[0].object.name == "SAP"


@pytest.mark.asyncio
async def test_extractor_tolerates_markdown_fenced_json() -> None:
    reply = (
        "```json\n"
        + json.dumps(
            {
                "triples": [
                    {
                        "subject": {"label": "Product", "name": "Atlas 5000", "properties": {}},
                        "relation": "HAS_FEATURE",
                        "object": {"label": "Feature", "name": "Dual-WAN", "properties": {}},
                    }
                ]
            }
        )
        + "\n```"
    )
    extractor = GraphExtractor(_settings(), completion_provider=ScriptedCompletionProvider(reply))

    triples = await extractor.extract(
        Chunk(
            chunk_id=UUID("00000000-0000-0000-0000-0000000000cc"),
            doc_id=UUID("00000000-0000-0000-0000-0000000000dd"),
            doc_title="Atlas 5000",
            doc_type="product_info",
            text="The Atlas 5000 has Dual-WAN.",
        )
    )

    assert len(triples) == 1
    assert triples[0].relation == "HAS_FEATURE"


@pytest.mark.asyncio
async def test_extractor_returns_empty_on_garbage() -> None:
    extractor = GraphExtractor(
        _settings(), completion_provider=ScriptedCompletionProvider("not json at all")
    )

    triples = await extractor.extract(_chunk("Some text."))

    assert triples == []


class FailingCompletionProvider:
    """Simulates Sarvam returning an empty answer for a chunk."""

    async def complete(self, *, system_prompt: str, user_prompt: str) -> CompletionResult:
        del system_prompt, user_prompt
        raise LLMError("Sarvam returned an empty answer.")


@pytest.mark.asyncio
async def test_extractor_tolerates_llm_failure_without_raising() -> None:
    # One chunk's LLM failure must not abort ingestion; it degrades to no
    # triples for that chunk (regression test for document-level crash).
    extractor = GraphExtractor(_settings(), completion_provider=FailingCompletionProvider())

    triples = await extractor.extract(_chunk("A heading with no facts."))

    assert triples == []


def test_allowed_edge_shapes() -> None:
    assert is_allowed_edge("Process", "HAS_STEP", "Step")
    assert is_allowed_edge("Product", "COMPATIBLE_WITH", "Product")
    assert not is_allowed_edge("Role", "HAS_STEP", "System")
    assert not is_allowed_edge("Process", "HAS_FEATURE", "Feature")
