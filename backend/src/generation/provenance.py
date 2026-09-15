"""Strict citation validation and provenance processing.

Phase 3 adds graph-derived facts as citable sources. Both vector chunks and
graph facts are normalized to a :class:`CitableSource` so the grounding logic
stays uniform: a claim survives only if it carries a ``[n]`` marker pointing at
a real source, and every kept claim gets a chunk-based citation (graph facts
cite the source chunk they were extracted from). A version warning is derived
when any cited source is a superseded SOP.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.api.schemas import Citation, CitationReference
from src.indexing.graph_schema import GraphFact
from src.indexing.vector_store import RetrievedChunk

_CITATION_RE = re.compile(r"\[(\d+)]")
_SEGMENT_RE = re.compile(r"(?<=[.!?])(?:\s+|$)|\n+")
_INSUFFICIENT_PREFIXES = (
    "insufficient context",
    "the provided context is insufficient",
    "i couldn't find",
    "i could not find",
)


class CitableSource(BaseModel):
    """A numbered source the synthesizer can cite, from a chunk or a graph fact.

    Both source kinds resolve to a chunk-based citation (decision: chunk-based
    citations via node ``chunk_ids``), so ``chunk_id`` is populated for graph
    facts from the fact's originating chunk when one is known.
    """

    text: str
    doc_title: str
    chunk_id: str | None = None
    sop_version: str | None = None
    is_current_version: bool = True
    origin: str = "internal_chunk"

    @classmethod
    def from_chunk(cls, retrieved: RetrievedChunk) -> CitableSource:
        chunk = retrieved.chunk
        return cls(
            text=chunk.text,
            doc_title=chunk.doc_title,
            chunk_id=str(chunk.chunk_id),
            sop_version=chunk.sop_version,
            is_current_version=chunk.is_current_version,
            origin="internal_chunk",
        )

    @classmethod
    def from_graph_fact(cls, fact: GraphFact) -> CitableSource:
        return cls(
            text=fact.as_text(),
            doc_title="Knowledge graph",
            chunk_id=fact.chunk_ids[0] if fact.chunk_ids else None,
            sop_version=None,
            is_current_version=True,
            origin="graph_fact",
        )


class GroundedAnswer(BaseModel):
    """Validated answer after unsupported claims have been removed."""

    answer: str
    citations: list[Citation]
    flagged_claims: list[str] = Field(default_factory=list)
    version_warning: str | None = None


def build_version_warning(citations: list[Citation]) -> str | None:
    """Return a warning if any cited internal source is a superseded SOP."""
    outdated = {
        citation.reference.doc_title
        for citation in citations
        if citation.reference.is_current_version is False
    }
    if not outdated:
        return None
    titles = ", ".join(sorted(outdated))
    return (
        f"Note: the answer references a superseded version of {titles}. "
        "A newer version of this document exists; verify against the current SOP."
    )


def ground_internal_answer(raw_answer: str, sources: list[CitableSource]) -> GroundedAnswer:
    """Keep only claims carrying valid ``[n]`` references to supplied chunks."""
    stripped = raw_answer.strip()
    if not stripped:
        return GroundedAnswer(
            answer="The provided context is insufficient to answer this question.",
            citations=[],
        )
    if stripped.casefold().startswith(_INSUFFICIENT_PREFIXES):
        return GroundedAnswer(
            answer="The provided context is insufficient to answer this question.",
            citations=[],
        )

    grounded_segments: list[str] = []
    citations: list[Citation] = []
    flagged: list[str] = []
    for segment in _segments(stripped):
        marker_numbers = [int(value) for value in _CITATION_RE.findall(segment)]
        valid_indices = list(dict.fromkeys(number - 1 for number in marker_numbers))
        if not valid_indices or any(index < 0 or index >= len(sources) for index in valid_indices):
            flagged.append(segment)
            continue

        grounded_segments.append(segment)
        claim_text = re.sub(r"\s+([.,!?;:])", r"\1", _CITATION_RE.sub("", segment)).strip()
        for source_index in valid_indices:
            source = sources[source_index]
            citations.append(
                Citation(
                    claim_text=claim_text,
                    source_tier="internal",
                    reference=CitationReference(
                        chunk_id=source.chunk_id,
                        doc_title=source.doc_title,
                        sop_version=source.sop_version,
                        is_current_version=source.is_current_version,
                    ),
                )
            )

    if not grounded_segments:
        return GroundedAnswer(
            answer="The provided context is insufficient to produce a citation-grounded answer.",
            citations=[],
            flagged_claims=flagged,
        )
    return GroundedAnswer(
        answer=" ".join(grounded_segments),
        citations=citations,
        flagged_claims=flagged,
        version_warning=build_version_warning(citations),
    )


def _segments(text: str) -> list[str]:
    return [segment.strip() for segment in _SEGMENT_RE.split(text) if segment.strip()]
