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
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from src.api.schemas import Citation, CitationReference
from src.indexing.graph_schema import GraphFact
from src.indexing.vector_store import RetrievedChunk
from src.retrieval.web_search import WebResult

# One or more adjacent citation markers, e.g. "[1]", "[1][2]", "[1, 2]" or "[1] [2]".
_CITATION_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*]")
_SINGLE_NUMBER_RE = re.compile(r"\d+")
# Split on sentence boundaries and hard newlines so markdown lists (one bullet
# per line) become individual segments rather than one giant blob.
_SEGMENT_RE = re.compile(r"(?<=[.!?])(?:\s+|$)|\n+")
# Leading markdown list/heading noise stripped from a segment before it becomes
# a clean, citable claim (e.g. "*   ", "- ", "1. ", "#### ").
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[*\-+]\s+|\d+[.)]\s+|#+\s+)")
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
    source_tier: Literal["internal", "external"] = "internal"
    url: str | None = None
    retrieved_at: datetime | None = None

    @classmethod
    def from_web(cls, result: WebResult) -> CitableSource:
        """Build an external source from a per-request Exa result."""
        return cls(
            text=result.snippet,
            doc_title=result.title,
            origin="web_result",
            source_tier="external",
            url=result.url,
            retrieved_at=result.retrieved_at,
        )

    def context_label(self) -> str:
        """Header for this source in the numbered synthesis context."""
        if self.source_tier == "external":
            return f"Web (external): {self.doc_title} ({self.url})"
        label = f"Internal document: {self.doc_title}"
        if self.sop_version:
            label += f" (version {self.sop_version})"
        if not self.is_current_version:
            label += " [SUPERSEDED - a newer version exists]"
        return label

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
    def from_graph_fact(
        cls,
        fact: GraphFact,
        *,
        chunk_lookup: dict[str, RetrievedChunk] | None = None,
    ) -> CitableSource:
        """Build a citable source from a graph fact.

        A graph fact is traceable to the chunk it was extracted from. When that
        chunk is among the retrieved sources, resolve the citation's real
        ``doc_title``/version/current-flag from it so a graph-derived claim cites
        the actual document rather than a synthetic "Knowledge graph" label.
        """
        chunk_id = fact.chunk_ids[0] if fact.chunk_ids else None
        source_chunk = chunk_lookup.get(chunk_id) if (chunk_lookup and chunk_id) else None
        if source_chunk is not None:
            chunk = source_chunk.chunk
            return cls(
                text=fact.as_text(),
                doc_title=chunk.doc_title,
                chunk_id=str(chunk.chunk_id),
                sop_version=chunk.sop_version,
                is_current_version=chunk.is_current_version,
                origin="graph_fact",
            )
        return cls(
            text=fact.as_text(),
            doc_title="Knowledge graph",
            chunk_id=chunk_id,
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


def ground_answer(raw_answer: str, sources: list[CitableSource]) -> GroundedAnswer:
    """Keep only claims carrying valid ``[n]`` references to supplied sources.

    Each citation takes its ``source_tier`` from the cited source, so internal
    chunks/graph facts and external web results stay distinguishable (SPEC 6.3).
    """
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
    seen_citations: set[tuple[str, str | None]] = set()
    for segment in _segments(stripped):
        valid_indices = _valid_source_indices(segment, len(sources))
        prose = _claim_text(segment)

        if not valid_indices:
            # A segment with no valid marker is only a "dropped claim" if it
            # actually asserts something. Pure structure (a bullet with no prose,
            # or stray markers the model put on their own line) is not a claim and
            # must not trip the uncited-claim rule or pollute flagged_claims.
            if prose:
                flagged.append(prose)
            continue

        grounded_segments.append(segment.strip())
        if not prose:
            # Markers with no prose (e.g. a line that is just "[1][2]"): keep them
            # in the readable answer but emit no empty-text citation.
            continue
        for source_index in valid_indices:
            source = sources[source_index]
            key = (prose, source.chunk_id or source.url or f"#{source_index}")
            if key in seen_citations:
                continue
            seen_citations.add(key)
            citations.append(_citation(prose, source))

    if not citations:
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


# Backwards-compatible name used by Phase 1-4 callers and tests.
ground_internal_answer = ground_answer


def _citation(prose: str, source: CitableSource) -> Citation:
    if source.source_tier == "external":
        reference = CitationReference(
            doc_title=source.doc_title,
            url=source.url,
            retrieved_at=source.retrieved_at,
        )
    else:
        reference = CitationReference(
            chunk_id=source.chunk_id,
            doc_title=source.doc_title,
            sop_version=source.sop_version,
            is_current_version=source.is_current_version,
        )
    return Citation(claim_text=prose, source_tier=source.source_tier, reference=reference)


def _valid_source_indices(segment: str, source_count: int) -> list[int]:
    """Return the 0-based source indices referenced by a segment's markers.

    Returns an empty list if the segment has no markers or any marker points
    outside the supplied source range (so a hallucinated ``[99]`` invalidates
    the reference rather than silently citing the wrong source).
    """
    numbers: list[int] = []
    for marker in _CITATION_RE.findall(segment):
        numbers.extend(int(value) for value in _SINGLE_NUMBER_RE.findall(marker))
    indices = list(dict.fromkeys(number - 1 for number in numbers))
    if not indices or any(index < 0 or index >= source_count for index in indices):
        return []
    return indices


def _claim_text(segment: str) -> str:
    """Strip markers and leading list/heading noise into a clean claim string."""
    without_prefix = _LIST_PREFIX_RE.sub("", segment)
    without_markers = _CITATION_RE.sub("", without_prefix)
    # Collapse space left before punctuation and any doubled whitespace.
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", without_markers)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    # Drop an empty parenthetical the markers left behind, e.g. "(document )".
    cleaned = re.sub(r"\(\s*(?:document|source|ref)?\s*\)", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" \t*-•").strip()


def _segments(text: str) -> list[str]:
    return [segment.strip() for segment in _SEGMENT_RE.split(text) if segment.strip()]
