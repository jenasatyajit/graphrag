"""Private GPU-hosted FastAPI service for BGE-M3 embeddings."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request

from src.api.config import Settings
from src.indexing.embeddings import (
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    InProcessBGEM3,
)


def create_embedding_service_app(
    settings: Settings | None = None,
    *,
    provider: EmbeddingProvider | None = None,
) -> FastAPI:
    """Build the private embedding service and load BGE-M3 once at startup."""
    active_settings = settings or Settings()
    if active_settings.embedding_service_token is None:
        raise ValueError("EMBEDDING_SERVICE_TOKEN is required by the embedding service.")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        embedder = provider or InProcessBGEM3(
            active_settings.embedding_model,
            device=active_settings.embedding_device,
            use_fp16=active_settings.embedding_use_fp16,
            expected_dimension=active_settings.embedding_dimension,
            inference_timeout_seconds=active_settings.embedding_inference_timeout_seconds,
        )
        if isinstance(embedder, InProcessBGEM3):
            embedder.load()
        app.state.embedder = embedder
        yield

    app = FastAPI(title="BGE-M3 Embedding Service", version="0.1.0", lifespan=lifespan)
    app.state.settings = active_settings

    def authorize(authorization: str | None) -> None:
        expected_secret = active_settings.embedding_service_token
        if expected_secret is None:
            return
        expected = f"Bearer {expected_secret.get_secret_value()}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Invalid embedding service token.")

    @app.get("/health")
    async def health(authorization: str | None = Header(default=None)) -> dict[str, str]:
        authorize(authorization)
        return {"status": "ok", "device": active_settings.embedding_device}

    @app.post("/embed", response_model=EmbeddingResponse)
    async def embed(
        payload: EmbeddingRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> EmbeddingResponse:
        authorize(authorization)
        if len(payload.texts) > active_settings.embedding_batch_size:
            raise HTTPException(
                status_code=422,
                detail=f"Embedding batch exceeds {active_settings.embedding_batch_size} texts.",
            )
        embedder: EmbeddingProvider = request.app.state.embedder
        result = await embedder.embed(payload.texts, mode=payload.mode)
        if len(result.dense) != len(payload.texts):
            raise HTTPException(
                status_code=502, detail="Embedding provider returned an invalid count."
            )
        return result

    return app
