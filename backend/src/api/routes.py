"""Minimal HTTP routes for the backend API."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from src.api.schemas import ComponentHealth, HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(request: Request) -> HealthResponse:
    """Return infrastructure dependency status without exposing credentials."""
    components: tuple[ComponentHealth, ComponentHealth] = await asyncio.gather(
        request.app.state.vector_store.verify_connectivity(),
        request.app.state.graph_store.verify_connectivity(),
    )
    status = "ok" if all(component.status == "ok" for component in components) else "degraded"
    return HealthResponse(status=status, components=list(components))
