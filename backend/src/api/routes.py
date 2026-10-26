"""Minimal HTTP routes for the backend API."""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    Request,
    Response,
    UploadFile,
    status,
)

from src.admin.store_reset import reset_stores
from src.api.errors import ConflictError, NotFoundError, ValidationError
from src.api.ingestion_task import run_ingestion_job
from src.api.schemas import (
    ComponentHealth,
    HealthResponse,
    IngestionSummary,
    JobStarted,
    JobStatusResponse,
    QueryRequest,
    QueryResponse,
    ResetRequest,
    ResetResponse,
)
from src.ingestion.parser import SUPPORTED_SUFFIXES

router = APIRouter()

_DOC_TYPES: frozenset[str] = frozenset({"SOP", "product_info"})


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
    """Delegate to the router agent; all business logic lives in ``src.agent``."""
    result = await request.app.state.agent.answer(payload.question, payload.conversation_id)
    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        used_web_search=result.used_web_search,
        version_warning=result.version_warning,
        tool_trace=result.tool_trace,
        conversation_id=result.conversation_id,
        notices=result.notices,
    )


@router.post("/ingest", response_model=IngestionSummary, tags=["ingestion"])
async def ingestion_status(request: Request) -> IngestionSummary:
    """Return the last GPU-hosted batch summary; never launch GPU work here."""
    return request.app.state.ingestion_status.latest()


# NOTE: the admin endpoints below are intentionally unauthenticated for the demo
# and are a known risk — they accept uploads and can wipe the data stores.


def _parse_effective_date(value: str | None) -> date | None:
    """Parse an optional ISO yyyy-mm-dd string, raising a 422 on bad input."""
    if value is None or value.strip() == "":
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValidationError(
            "effective_date must be an ISO date (yyyy-mm-dd).",
        ) from exc


@router.post(
    "/documents",
    response_model=JobStarted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["admin"],
)
async def ingest_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File()],
    doc_type: Annotated[str, Form()],
    sop_version: Annotated[str | None, Form()] = None,
    doc_title: Annotated[str | None, Form()] = None,
    effective_date: Annotated[str | None, Form()] = None,
) -> JobStarted:
    """Accept one document upload and start an in-process ingestion job.

    Validates the file type and metadata, rejects a second job while one is
    active (409), stores the upload in a per-job temp directory, and schedules
    the GPU-bound ingestion in the background. Returns 202 immediately.
    """
    settings = request.app.state.settings
    job_store = request.app.state.job_store

    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValidationError(
            f"Unsupported file type '{suffix or filename}'. Allowed: {supported}."
        )
    if doc_type not in _DOC_TYPES:
        allowed = ", ".join(sorted(_DOC_TYPES))
        raise ValidationError(f"doc_type must be one of: {allowed}.")
    parsed_effective_date = _parse_effective_date(effective_date)

    payload = await file.read()
    if len(payload) > settings.max_upload_bytes:
        raise ValidationError(
            f"File exceeds the maximum upload size of {settings.max_upload_bytes} bytes."
        )
    if not payload:
        raise ValidationError("The uploaded file is empty.")

    job_id = str(uuid4())
    job = await job_store.try_begin(job_id=job_id, filename=filename, doc_type=doc_type)
    if job is None:
        raise ConflictError("An ingestion job is already running. Try again when it finishes.")

    job_dir = Path(settings.ingestion_upload_temp_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / Path(filename).name).write_bytes(payload)

    background_tasks.add_task(
        run_ingestion_job,
        job_store,
        settings,
        job_id=job_id,
        job_dir=job_dir,
        doc_type=doc_type,
        sop_version=sop_version,
        doc_title=doc_title,
        effective_date=parsed_effective_date,
    )
    return JobStarted(job_id=job_id, status="pending")


@router.get(
    "/documents/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["admin"],
)
async def ingestion_job_status(job_id: str, request: Request) -> JobStatusResponse:
    """Return the current status of an ingestion job (404 if unknown)."""
    job = await request.app.state.job_store.get(job_id)
    if job is None:
        raise NotFoundError(f"No ingestion job with id '{job_id}'.")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        filename=job.filename,
        doc_type=job.doc_type,
        created_at=job.created_at,
        updated_at=job.updated_at,
        summary=job.summary,
        error=job.error,
    )


@router.post("/admin/reset", response_model=ResetResponse, tags=["admin"])
async def admin_reset(payload: ResetRequest, request: Request) -> ResetResponse:
    """Wipe the selected data stores. Requires ``confirm=true`` to proceed."""
    if not payload.confirm:
        raise ValidationError("reset requires confirm=true to proceed.")
    results = await reset_stores(request.app.state.settings, scope=payload.scope)
    return ResetResponse(scope=payload.scope, results=results)
