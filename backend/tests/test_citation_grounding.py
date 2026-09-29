"""Citation-grounding and synthesis enforcement tests."""

from types import SimpleNamespace
from uuid import UUID

import pytest
from sarvamai.core.api_error import ApiError

from src.api.config import Settings
from src.generation.provenance import CitableSource, ground_internal_answer
from src.generation.synthesizer import (
    AnswerSynthesizer,
    CompletionResult,
    SarvamCompletionProvider,
)
from src.indexing.vector_store import RetrievedChunk
from src.ingestion.chunker import Chunk


def _source() -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=UUID("00000000-0000-0000-0000-000000000030"),
            doc_id=UUID("00000000-0000-0000-0000-000000000031"),
            doc_title="Refund SOP",
            doc_type="SOP",
            text="Refunds require manager approval.",
            sop_version="v3.0",
        ),
        score=0.9,
    )


def _citable() -> CitableSource:
    return CitableSource.from_chunk(_source())


def test_uncited_claim_is_removed_and_flagged() -> None:
    result = ground_internal_answer(
        "Refunds require manager approval [1]. This policy never expires.",
        [_citable()],
    )

    assert result.answer == "Refunds require manager approval [1]."
    assert len(result.citations) == 1
    assert result.citations[0].reference.chunk_id == str(_source().chunk.chunk_id)
    assert result.flagged_claims == ["This policy never expires."]


def _two_sources() -> list[CitableSource]:
    a = CitableSource.from_chunk(_source())
    b = CitableSource(
        text="Finance approval requires SAP.",
        doc_title="Expense Approval SOP",
        chunk_id="00000000-0000-0000-0000-0000000000ff",
        sop_version="v3.0",
        is_current_version=True,
    )
    return [a, b]


def test_markdown_bullet_list_is_grounded_not_shredded() -> None:
    # A markdown bullet list with a marker per line must survive as clean prose,
    # not be discarded as uncited fragments (regression for the "[1][2]" answers).
    raw = "* Refunds require manager approval [1].\n* Finance approval requires SAP [2]."
    result = ground_internal_answer(raw, _two_sources())

    assert "Refunds require manager approval [1]." in result.answer
    assert "Finance approval requires SAP [2]." in result.answer
    assert len(result.citations) == 2
    # List bullets are stripped from the citable claim text.
    assert result.citations[0].claim_text == "Refunds require manager approval."
    assert result.citations[1].claim_text == "Finance approval requires SAP."
    assert result.flagged_claims == []


def test_adjacent_and_comma_markers_resolve_all_sources() -> None:
    result = ground_internal_answer("Both sources agree [1][2].", _two_sources())
    assert len(result.citations) == 2

    comma = ground_internal_answer("Both sources agree [1, 2].", _two_sources())
    assert len(comma.citations) == 2


def test_inline_document_marker_leaves_clean_claim_text() -> None:
    # "(document [1])" phrasing must not leave an empty "(document )" behind.
    result = ground_internal_answer(
        "The Expense Approval SOP (document [1]) states managers approve expenses [1].",
        _two_sources(),
    )
    assert "(document )" not in result.citations[0].claim_text
    assert "document" not in result.citations[0].claim_text.lower()


def test_marker_only_line_is_kept_without_empty_citation() -> None:
    # A line that is only markers keeps the answer readable but emits no
    # empty-text citation, and does not count as a dropped claim.
    result = ground_internal_answer(
        "Managers approve expenses [1].\n[1][2]",
        _two_sources(),
    )
    assert result.citations
    assert all(citation.claim_text for citation in result.citations)
    assert result.flagged_claims == []


def test_graph_fact_cites_real_document_when_chunk_known() -> None:
    from src.indexing.graph_schema import GraphFact

    source = _source()
    lookup = {str(source.chunk.chunk_id): source}
    fact = GraphFact(
        subject_name="Finance Approval",
        relation="REQUIRES",
        object_name="SAP",
        chunk_ids=[str(source.chunk.chunk_id)],
    )

    citable = CitableSource.from_graph_fact(fact, chunk_lookup=lookup)

    assert citable.doc_title == "Refund SOP"
    assert citable.doc_title != "Knowledge graph"
    assert citable.sop_version == "v3.0"


class FakeCompletionProvider:
    async def complete(self, *, system_prompt: str, user_prompt: str) -> CompletionResult:
        assert "ONLY" in system_prompt
        assert "[1] Document: Refund SOP" in user_prompt
        return CompletionResult(text="Refunds require manager approval [1].")


@pytest.mark.asyncio
async def test_synthesizer_builds_numbered_context_and_citations() -> None:
    result = await AnswerSynthesizer(FakeCompletionProvider()).synthesize(
        "Who approves refunds?",
        [_source()],
    )

    assert result.answer.endswith("[1].")
    assert result.citations[0].claim_text == "Refunds require manager approval."
    assert result.citations[0].source_tier == "internal"


def test_refusal_prefix_cannot_bypass_uncited_claim_filter() -> None:
    result = ground_internal_answer(
        "Insufficient context. The CEO approved the exception.",
        [_citable()],
    )

    assert result.answer == "The provided context is insufficient to answer this question."
    assert result.citations == []


class RetryChat:
    def __init__(self) -> None:
        self.attempts = 0

    async def completions(self, **kwargs: object) -> SimpleNamespace:
        del kwargs
        self.attempts += 1
        if self.attempts == 1:
            raise ApiError(status_code=503, body="temporary")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Grounded [1]."))],
            usage=None,
        )


@pytest.mark.asyncio
async def test_sarvam_provider_retries_transient_server_error(settings: Settings) -> None:
    chat = RetryChat()
    provider = SarvamCompletionProvider(settings, client=SimpleNamespace(chat=chat))

    result = await provider.complete(system_prompt="system", user_prompt="user")

    assert result.text == "Grounded [1]."
    assert chat.attempts == 2
