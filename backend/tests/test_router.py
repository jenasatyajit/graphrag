"""Phase 1 dense-RAG orchestration tests; Phase 5 adds routing cases."""

import pytest

from src.agent.router import NaiveRagService
from src.generation.synthesizer import AnswerSynthesizer
from tests.conftest import FakeCompletionProvider, FakeEmbeddingProvider, FakeVectorStore


@pytest.mark.asyncio
async def test_naive_rag_service_runs_embed_retrieve_synthesize() -> None:
    service = NaiveRagService(
        FakeEmbeddingProvider(),
        FakeVectorStore(),  # type: ignore[arg-type]
        AnswerSynthesizer(FakeCompletionProvider()),
    )

    result = await service.query("Who approves large expenses?")

    assert result.answer == "Managers approve expenses above $500 [1]."
    assert len(result.citations) == 1
