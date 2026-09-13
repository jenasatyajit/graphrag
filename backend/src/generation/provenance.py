"""Strict Phase 1 citation validation and provenance processing."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.api.schemas import Citation, CitationReference
from src.indexing.vector_store import RetrievedChunk

_CITATION_RE = re.compile(r"\[(\d+)]")
_SEGMENT_RE = re.compile(r"(?<=[.!?])(?:\s+|$)|\n+")
_INSUFFICIENT_PREFIXES = (
    "insufficient context",
    "the provided context is insufficient",
    "i couldn't find",
    "i could not find",
)


class GroundedAnswer(BaseModel):
    """Validated answer after unsupported claims have been removed."""

    answer: str
    citations: list[Citation]
    flagged_claims: list[str] = Field(default_factory=list)


def ground_internal_answer(raw_answer: str, sources: list[RetrievedChunk]) -> GroundedAnswer:
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
            chunk = sources[source_index].chunk
            citations.append(
                Citation(
                    claim_text=claim_text,
                    source_tier="internal",
                    reference=CitationReference(
                        chunk_id=str(chunk.chunk_id),
                        doc_title=chunk.doc_title,
                        sop_version=chunk.sop_version,
                        is_current_version=chunk.is_current_version,
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
    )


def _segments(text: str) -> list[str]:
    return [segment.strip() for segment in _SEGMENT_RE.split(text) if segment.strip()]
