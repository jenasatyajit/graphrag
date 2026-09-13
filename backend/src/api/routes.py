"""Minimal HTTP routes for the backend API."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request, Response, status

from src.api.schemas import (
    ComponentHealth,
    HealthResponse,
    IngestionSummary,
    QueryRequest,
    QueryResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(request: Request, response: Response) -> HealthResponse:
    """Report Phase 1 required dependencies and pending Phase 3 graph health."""
    components: list[ComponentHealth] = list(
        await asyncio.gather(
            request.app.state.vector_store.verify_connectivity(),
            request.app.state.graph_store.verify_connectivity(),
            request.app.state.embedding_health.verify_connectivity(),
        )
    )
    required = {item.name: item.status for item in components}
    ready = required.get("qdrant") == "ok" and required.get("embedding_service") == "ok"
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    overall = "ok" if all(item.status == "ok" for item in components) else "degraded"
    return HealthResponse(status=overall, components=components)


@router.post("/query", response_model=QueryResponse, tags=["query"])
async def query(payload: QueryRequest, request: Request) -> QueryResponse:
    """Call the Phase 1 RAG service; routing arrives in Phase 5."""
    grounded = await request.app.state.query_service.query(payload.question)
    return QueryResponse(
        answer=grounded.answer,
        citations=grounded.citations,
        used_web_search=False,
        version_warning=None,
        tool_trace=["internal_search"],
    )


@router.post("/ingest", response_model=IngestionSummary, tags=["ingestion"])
async def ingestion_status(request: Request) -> IngestionSummary:
    """Return the last GPU-hosted batch summary; never launch GPU work here."""
    return request.app.state.ingestion_status.latest()
