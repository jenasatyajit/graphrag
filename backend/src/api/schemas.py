"""Pydantic schemas shared by the API boundary."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ComponentHealth(BaseModel):
    """Health report for an infrastructure dependency."""

    name: Literal["qdrant", "neo4j", "embedding_service"]
    status: Literal["ok", "pending", "unhealthy"]
    detail: str | None = None


class HealthResponse(BaseModel):
    """Response body for the service health endpoint."""

    status: Literal["ok", "degraded"]
    components: list[ComponentHealth]


class CitationReference(BaseModel):
    """Traceable source metadata for an answer citation."""

    chunk_id: str | None = None
    doc_title: str
    sop_version: str | None = None
    is_current_version: bool | None = None
    url: str | None = None
    retrieved_at: datetime | None = None


class Citation(BaseModel):
    """A factual claim and the source that supports it."""

    claim_text: str
    source_tier: Literal["internal", "external"]
    reference: CitationReference


class QueryRequest(BaseModel):
    """Question submitted to the company-knowledge pipeline."""

    question: str = Field(min_length=1, max_length=10_000)
    conversation_id: str | None = Field(default=None, max_length=200)


class QueryResponse(BaseModel):
    """SPEC 8 response, plus the server-issued ``conversation_id`` and notices.

    ``conversation_id`` must be sent back on the next request to continue the
    conversation (deviation D-15). ``notices`` carries non-claim status messages,
    e.g. that live web search was unavailable.
    """

    answer: str
    citations: list[Citation]
    used_web_search: bool = False
    version_warning: str | None = None
    tool_trace: list[str]
    conversation_id: str
    notices: list[str] = Field(default_factory=list)


class IngestionSummary(BaseModel):
    """Machine-readable result from the GPU-hosted ingestion batch."""

    docs_processed: int = Field(ge=0)
    chunks_created: int = Field(ge=0)
    tables_extracted: int = Field(default=0, ge=0)
    images_extracted: int = Field(default=0, ge=0)
    graph_nodes_created: int = Field(default=0, ge=0)
    graph_edges_created: int = Field(default=0, ge=0)
    failed_documents: list[str] = Field(default_factory=list)
    completed_at: datetime


class ErrorPayload(BaseModel):
    """Stable user-facing error payload."""

    code: str
    message: str
    details: Any | None = None
    request_id: str


class ErrorEnvelope(BaseModel):
    """Top-level error response envelope."""

    error: ErrorPayload
