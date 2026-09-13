"""Pydantic schemas shared by the API boundary."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class ComponentHealth(BaseModel):
    """Health report for an infrastructure dependency."""

    name: Literal["qdrant", "neo4j"]
    status: Literal["ok", "pending", "unhealthy"]
    detail: str | None = None


class HealthResponse(BaseModel):
    """Response body for the service health endpoint."""

    status: Literal["ok", "degraded"]
    components: list[ComponentHealth]


class ErrorPayload(BaseModel):
    """Stable user-facing error payload."""

    code: str
    message: str
    details: Any | None = None
    request_id: str


class ErrorEnvelope(BaseModel):
    """Top-level error response envelope."""

    error: ErrorPayload
