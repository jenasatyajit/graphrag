"""Tests for API construction, health, errors, and Phase 1 query flow."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.config import Settings
from src.api.errors import NotFoundError
from src.api.main import create_app


def test_create_app_and_health_check(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    # Phase 3: the graph store reports a real "ok" status, so with all fake
    # dependencies healthy the aggregate health is "ok".
    assert response.json()["status"] == "ok"
    assert {item["name"] for item in response.json()["components"]} == {
        "qdrant",
        "neo4j",
        "embedding_service",
    }
    neo4j = next(c for c in response.json()["components"] if c["name"] == "neo4j")
    assert neo4j["status"] == "ok"


def test_post_query_returns_spec_response_shape(client: TestClient) -> None:
    response = client.post(
        "/query",
        json={"question": "Who approves large expenses?", "conversation_id": None},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Managers approve expenses above $500 [1]."
    assert payload["citations"][0]["source_tier"] == "internal"
    assert payload["citations"][0]["reference"]["doc_title"] == "Expense Approval SOP"
    assert payload["used_web_search"] is False
    assert payload["version_warning"] is None
    assert payload["tool_trace"] == ["internal_search"]
    assert payload["conversation_id"]
    assert payload["notices"] == []


def test_request_id_is_generated_and_echoed(client: TestClient) -> None:
    response = client.get("/health")
    assert response.headers["X-Request-ID"]


def test_caller_request_id_is_preserved(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "request-123"})
    assert response.headers["X-Request-ID"] == "request-123"


def test_domain_error_uses_uniform_envelope(settings: Settings) -> None:
    app: FastAPI = create_app(settings)

    @app.get("/test-not-found")
    async def raise_not_found() -> None:
        raise NotFoundError("The requested test resource does not exist.")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/test-not-found", headers={"X-Request-ID": "error-123"})

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "not_found",
            "message": "The requested test resource does not exist.",
            "details": None,
            "request_id": "error-123",
        }
    }
