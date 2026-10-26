"""Tests for the admin endpoints: async ingestion upload and store reset.

All vendor integrations are faked: the GPU-bound ``run_ingestion`` is patched
with an in-process fake, and the store-reset helpers are monkeypatched, so these
tests never import Docling/BGE-M3 or touch live Qdrant/Neo4j. The import-boundary
test additionally asserts that ``src.api.main`` stays CPU-safe.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from src.agent.memory import InMemoryConversationStore
from src.api.jobs import JobStore
from src.api.main import create_app
from src.api.schemas import IngestionSummary, ResetStoreOutcome
from tests.conftest import (
    FakeCompletionProvider,
    FakeEmbeddingProvider,
    FakeGraphStore,
    FakeReranker,
    FakeVectorStore,
    FakeWebSearch,
    ScriptedRouterLLM,
    build_client,
    make_settings,
    tool_decision,
)


def _fake_wiring() -> dict[str, Any]:
    """Keyword overrides for ``create_app`` using the shared test fakes."""
    return {
        "vector_store": cast(Any, FakeVectorStore()),
        "graph_store": cast(Any, FakeGraphStore()),
        "embedding_provider": FakeEmbeddingProvider(),
        "completion_provider": FakeCompletionProvider(),
        "reranker": FakeReranker(),
        "router_llm": ScriptedRouterLLM(tool_decision("internal_search")),
        "web_searcher": FakeWebSearch(),
        "conversation_store": InMemoryConversationStore(max_turns=20, ttl=timedelta(hours=1)),
    }


_SUMMARY = IngestionSummary(
    docs_processed=1,
    chunks_created=7,
    tables_extracted=2,
    images_extracted=0,
    graph_nodes_created=3,
    graph_edges_created=4,
    failed_documents=[],
    completed_at=datetime(2026, 1, 1, tzinfo=UTC),
)


@pytest.fixture
def admin_client(tmp_path: Path) -> TestClient:
    settings = make_settings(
        ingestion_upload_temp_dir=tmp_path / "uploads",
        ingestion_data_dir=tmp_path / "raw",
    )
    return build_client(settings=settings)


def _fake_run_ingestion_factory(summary: IngestionSummary):
    async def _fake_run_ingestion(settings: Any, **kwargs: Any) -> IngestionSummary:
        del settings, kwargs
        return summary

    return _fake_run_ingestion


def test_upload_creates_job_and_succeeds(
    admin_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "scripts.run_ingestion.run_ingestion",
        _fake_run_ingestion_factory(_SUMMARY),
    )
    resp = admin_client.post(
        "/documents",
        files={"file": ("policy.md", b"# hello world", "text/markdown")},
        data={"doc_type": "SOP", "sop_version": "v1", "doc_title": "Policy"},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "pending"
    job_id = body["job_id"]
    assert job_id

    # TestClient runs BackgroundTasks synchronously, so the job is terminal now.
    status_resp = admin_client.get(f"/documents/jobs/{job_id}")
    assert status_resp.status_code == 200
    status_body = status_resp.json()
    assert status_body["status"] == "succeeded"
    assert status_body["summary"]["chunks_created"] == 7
    assert status_body["error"] is None


def test_upload_transitions_to_failed_on_error(
    admin_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _boom(settings: Any, **kwargs: Any) -> IngestionSummary:
        del settings, kwargs
        raise RuntimeError("ingestion blew up")

    monkeypatch.setattr("scripts.run_ingestion.run_ingestion", _boom)
    resp = admin_client.post(
        "/documents",
        files={"file": ("policy.md", b"# hello", "text/markdown")},
        data={"doc_type": "product_info"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    status_body = admin_client.get(f"/documents/jobs/{job_id}").json()
    assert status_body["status"] == "failed"
    assert status_body["error"]


def test_upload_rejects_bad_suffix(admin_client: TestClient) -> None:
    resp = admin_client.post(
        "/documents",
        files={"file": ("notes.txt", b"nope", "text/plain")},
        data={"doc_type": "SOP"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_upload_rejects_bad_doc_type(admin_client: TestClient) -> None:
    resp = admin_client.post(
        "/documents",
        files={"file": ("policy.md", b"# hi", "text/markdown")},
        data={"doc_type": "invalid"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_second_upload_while_busy_returns_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(ingestion_upload_temp_dir=tmp_path / "uploads")

    started = {"first": False}

    async def _never_finishes(settings: Any, **kwargs: Any) -> IngestionSummary:
        del settings, kwargs
        started["first"] = True
        # Simulate an in-flight job that the background task has not resolved yet.
        raise AssertionError("should not be called in this test")

    monkeypatch.setattr("scripts.run_ingestion.run_ingestion", _never_finishes)

    # Inject a job directly into the store to simulate a busy GPU VM, then
    # confirm a second upload is rejected with 409 before any task runs.
    import asyncio

    store = JobStore()
    asyncio.run(store.try_begin(job_id="busy", filename="x.md", doc_type="SOP"))
    app = create_app(settings=settings, **_fake_wiring())
    app.state.job_store = store
    client = TestClient(app)

    resp = client.post(
        "/documents",
        files={"file": ("policy.md", b"# hi", "text/markdown")},
        data={"doc_type": "SOP"},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "conflict"


def test_job_status_unknown_returns_404(admin_client: TestClient) -> None:
    resp = admin_client.get("/documents/jobs/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_reset_requires_confirm(admin_client: TestClient) -> None:
    resp = admin_client.post("/admin/reset", json={"scope": "all", "confirm": False})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_reset_calls_helpers_with_scope(
    admin_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def _fake_reset_qdrant(settings: Any) -> ResetStoreOutcome:
        del settings
        calls.append("qdrant")
        return ResetStoreOutcome(store="qdrant", status="reset", detail="ok")

    async def _fake_reset_neo4j(settings: Any) -> ResetStoreOutcome:
        del settings
        calls.append("neo4j")
        return ResetStoreOutcome(store="neo4j", status="reset", detail="ok")

    monkeypatch.setattr("src.admin.store_reset.reset_qdrant", _fake_reset_qdrant)
    monkeypatch.setattr("src.admin.store_reset.reset_neo4j", _fake_reset_neo4j)

    resp = admin_client.post("/admin/reset", json={"scope": "qdrant", "confirm": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["scope"] == "qdrant"
    assert [r["store"] for r in body["results"]] == ["qdrant"]
    assert calls == ["qdrant"]


def test_main_imports_without_gpu_dependencies() -> None:
    """src.api.main must import cleanly without pulling in GPU-only libraries."""
    import src.api.main  # noqa: F401

    for module in ("torch", "FlagEmbedding", "flagembedding", "docling"):
        assert module not in sys.modules, f"{module} must not be imported by src.api.main"
