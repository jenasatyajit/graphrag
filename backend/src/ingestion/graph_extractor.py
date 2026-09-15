"""Schema-constrained LLM extraction of graph triples from chunks (Phase 3).

The extractor sends one chunk at a time to Sarvam (reasoning disabled) with the
schema inlined in the prompt, parses the JSON response, and validates every
triple against :mod:`src.indexing.graph_schema`. Off-schema triples are dropped
rather than persisted, so the graph can never contain labels or relationships
outside SPEC 6.2. ``Process`` nodes are enriched with the source document's
title/version so the SOP ``SUPERSEDES`` chain can be built during ingestion.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from pydantic import ValidationError

from src.agent.prompts import GRAPH_EXTRACTION_SYSTEM_PROMPT
from src.api.config import Settings
from src.generation.synthesizer import CompletionProvider, SarvamCompletionProvider
from src.indexing.graph_schema import GraphNode, GraphTriple
from src.ingestion.chunker import Chunk

logger = structlog.get_logger(__name__)


class GraphExtractor:
    """Turn a chunk's text into validated, schema-compliant graph triples."""

    def __init__(
        self,
        settings: Settings,
        *,
        completion_provider: CompletionProvider | None = None,
    ) -> None:
        self._provider = completion_provider or SarvamCompletionProvider(
            settings,
            reasoning_effort=settings.llm_extract_reasoning_effort,
            max_tokens=settings.llm_extract_max_tokens,
        )

    async def extract(self, chunk: Chunk) -> list[GraphTriple]:
        """Extract validated triples for one chunk.

        Never raises on a malformed or off-schema model response: a bad chunk
        yields an empty list and is logged, so one noisy chunk cannot abort the
        whole ingestion batch.
        """
        completion = await self._provider.complete(
            system_prompt=GRAPH_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=self._build_user_prompt(chunk),
        )
        raw_triples = _parse_triples_payload(completion.text)
        triples: list[GraphTriple] = []
        dropped = 0
        for raw in raw_triples:
            triple = self._validate_triple(raw, chunk)
            if triple is None:
                dropped += 1
                continue
            triples.append(triple)
        logger.info(
            "graph_triples_extracted",
            chunk_id=str(chunk.chunk_id),
            extracted=len(triples),
            dropped=dropped,
        )
        return triples

    @staticmethod
    def _build_user_prompt(chunk: Chunk) -> str:
        header = f"Document: {chunk.doc_title} (type: {chunk.doc_type})"
        return f"{header}\n\nChunk:\n{chunk.text.strip()}"

    def _validate_triple(self, raw: Any, chunk: Chunk) -> GraphTriple | None:
        """Validate one raw triple, enriching Process nodes for versioning."""
        if not isinstance(raw, dict):
            return None
        try:
            triple = GraphTriple.model_validate(raw)
        except ValidationError:
            return None
        triple.subject = self._enrich_node(triple.subject, chunk)
        triple.object = self._enrich_node(triple.object, chunk)
        return triple

    @staticmethod
    def _enrich_node(node: GraphNode, chunk: Chunk) -> GraphNode:
        """Attach SOP-versioning properties to Process nodes.

        ``doc_title``/``sop_version``/``is_current_version`` let the graph store
        build the ``SUPERSEDES`` chain and flag superseded versions. Other node
        types are returned unchanged.
        """
        if node.label != "Process":
            return node
        properties = dict(node.properties)
        properties["doc_title"] = chunk.doc_title
        if chunk.sop_version is not None:
            properties["sop_version"] = chunk.sop_version
        properties["is_current_version"] = chunk.is_current_version
        return node.model_copy(update={"properties": properties})


def _parse_triples_payload(text: str) -> list[Any]:
    """Extract the ``triples`` list from a model response, tolerating noise.

    Handles bare JSON, JSON wrapped in markdown fences, and surrounding prose by
    locating the outermost JSON object. Returns an empty list on any failure.
    """
    candidate = _strip_code_fences(text).strip()
    obj = _load_json_object(candidate)
    if obj is None:
        return []
    triples = obj.get("triples")
    return triples if isinstance(triples, list) else []


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Drop the opening fence (optionally ```json) and a trailing fence.
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return text


def _load_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        pass
    # Fall back to the first balanced ``{...}`` span if the model added prose.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None
