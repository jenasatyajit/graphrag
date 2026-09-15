"""Query-time knowledge-graph retrieval (Phase 3).

Extracts candidate entity names from the question with an LLM (deviation D-9:
LLM-based rather than local NER, reasoning disabled for speed), fuzzy-matches
them to graph node names, traverses 1-2 hops, and returns connected facts as
citable triples. Graph retrieval is a quality booster, not a hard dependency:
any failure (extraction, matching, or traversal) degrades to an empty result
with a logged warning so the vector-only path still answers the question.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from src.agent.prompts import ENTITY_EXTRACTION_SYSTEM_PROMPT
from src.generation.synthesizer import CompletionProvider
from src.indexing.graph_schema import GraphFact
from src.indexing.graph_store import GraphStore

logger = structlog.get_logger(__name__)


class GraphSearchService:
    """Resolve question entities to graph nodes and return neighborhood facts."""

    def __init__(
        self,
        graph_store: GraphStore,
        entity_extractor: CompletionProvider,
    ) -> None:
        self._graph_store = graph_store
        self._entity_extractor = entity_extractor

    async def search(self, question: str) -> list[GraphFact]:
        """Return graph facts relevant to the question, or [] on any failure."""
        try:
            entities = await self._extract_entities(question)
            if not entities:
                logger.info("graph_search_no_entities", question_length=len(question))
                return []
            matched = await self._graph_store.match_entities(entities)
            if not matched:
                logger.info(
                    "graph_search_no_matches",
                    extracted_entities=entities,
                )
                return []
            facts = await self._graph_store.neighborhood_facts(matched)
            logger.info(
                "graph_search_completed",
                extracted_entities=entities,
                matched_nodes=matched,
                fact_count=len(facts),
            )
            return facts
        except Exception:
            # Degrade gracefully: a graph outage must not break retrieval.
            logger.warning("graph_search_failed", exc_info=True)
            return []

    async def _extract_entities(self, question: str) -> list[str]:
        completion = await self._entity_extractor.complete(
            system_prompt=ENTITY_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=question.strip(),
        )
        return _parse_entities_payload(completion.text)


def _parse_entities_payload(text: str) -> list[str]:
    """Extract the ``entities`` string list from a model response."""
    candidate = _strip_code_fences(text).strip()
    obj = _load_json_object(candidate)
    if obj is None:
        return []
    entities = obj.get("entities")
    if not isinstance(entities, list):
        return []
    result: list[str] = []
    for item in entities:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return list(dict.fromkeys(result))


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
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
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None
