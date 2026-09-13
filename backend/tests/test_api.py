"""Tests for Phase 0 API construction and common HTTP behavior."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.config import Settings
from src.api.errors import NotFoundError
from src.api.main import create_app


def test_create_app_and_health_check(client: TestClient) -> None:
    """The Phase 0 health endpoint should expose pending dependency states."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert {item["name"] for item in response.json()["components"]} == {"qdrant", "neo4j"}


def test_request_id_is_generated_and_echoed(client: TestClient) -> None:
    """Every response should include the request correlation identifier."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]


def test_caller_request_id_is_preserved(client: TestClient) -> None:
    """A caller-provided request ID should be echoed unchanged."""
    response = client.get("/health", headers={"X-Request-ID": "request-123"})

    assert response.headers["X-Request-ID"] == "request-123"


def test_domain_error_uses_uniform_envelope(settings: Settings) -> None:
    """Domain errors should be safe, typed, and correlated to the request."""
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
