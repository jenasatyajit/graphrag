"""Application exceptions and uniform FastAPI error responses."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.schemas import ErrorEnvelope, ErrorPayload

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for safe, user-facing application errors."""

    code = "application_error"
    status_code = 500

    def __init__(self, message: str, *, details: Any | None = None) -> None:
        super().__init__(message)
        self.details = details


class RetrievalError(AppError):
    """Raised when vector or hybrid retrieval fails."""

    code = "retrieval_error"
    status_code = 503


class GraphError(AppError):
    """Raised when graph access fails."""

    code = "graph_error"
    status_code = 503


class RerankError(AppError):
    """Raised when reranking cannot complete."""

    code = "rerank_error"
    status_code = 503


class LLMError(AppError):
    """Raised when a language-model operation fails."""

    code = "llm_error"
    status_code = 503


class IngestionError(AppError):
    """Raised when batch ingestion cannot complete."""

    code = "ingestion_error"
    status_code = 500


class NotFoundError(AppError):
    """Raised when a requested resource does not exist."""

    code = "not_found"
    status_code = 404


class ValidationError(AppError):
    """Raised for domain-level validation failures."""

    code = "validation_error"
    status_code = 422


class ConflictError(AppError):
    """Raised when a request conflicts with current server state (e.g. a busy job)."""

    code = "conflict"
    status_code = 409


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _error_response(
    *,
    code: str,
    message: str,
    details: Any | None,
    request_id: str,
    status_code: int,
) -> JSONResponse:
    payload = ErrorEnvelope(
        error=ErrorPayload(
            code=code,
            message=message,
            details=details,
            request_id=request_id,
        )
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def register_exception_handlers(app: FastAPI) -> None:
    """Register safe, consistently shaped error responses for the API."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning("application_error", exc_info=exc)
        return _error_response(
            code=exc.code,
            message=str(exc),
            details=exc.details,
            request_id=_request_id(request),
            status_code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        details: Sequence[dict[str, Any]] = exc.errors()
        return _error_response(
            code="request_validation_error",
            message="The request body is invalid.",
            details=list(details),
            request_id=_request_id(request),
            status_code=422,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unexpected_error")
        return _error_response(
            code="internal_error",
            message="An unexpected error occurred.",
            details=None,
            request_id=_request_id(request),
            status_code=500,
        )
