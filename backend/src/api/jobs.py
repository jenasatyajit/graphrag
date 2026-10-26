"""In-process ingestion job store.

Tracks the lifecycle of asynchronous document-ingestion jobs started by
``POST /documents``. This store is intentionally in-process and lock-guarded:
it holds a single-active-job invariant (the GPU runs one ingestion at a time)
and is attached to ``app.state`` like the other services. Jobs are kept in
memory only, so **all job records are lost on application restart** — this is a
deliberate simplification for the demo, not a bug. Nothing GPU-bound is imported
here; this module stays CPU-safe.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.api.schemas import IngestionSummary, JobStatusValue

_ACTIVE_STATUSES: frozenset[str] = frozenset({"pending", "running"})


@dataclass
class IngestionJob:
    """Mutable record describing one ingestion job's lifecycle."""

    job_id: str
    status: JobStatusValue
    filename: str
    doc_type: str
    created_at: datetime
    updated_at: datetime
    summary: IngestionSummary | None = None
    error: str | None = None


@dataclass
class JobStore:
    """Async-lock-guarded registry of ingestion jobs (lost on restart)."""

    _jobs: dict[str, IngestionJob] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def try_begin(self, *, job_id: str, filename: str, doc_type: str) -> IngestionJob | None:
        """Register a new pending job, or return ``None`` if one is already active.

        Enforces the single-active-job invariant atomically under the lock so two
        concurrent uploads cannot both start a job.
        """
        async with self._lock:
            if any(job.status in _ACTIVE_STATUSES for job in self._jobs.values()):
                return None
            now = datetime.now(UTC)
            job = IngestionJob(
                job_id=job_id,
                status="pending",
                filename=filename,
                doc_type=doc_type,
                created_at=now,
                updated_at=now,
            )
            self._jobs[job_id] = job
            return job

    async def get(self, job_id: str) -> IngestionJob | None:
        """Return the job with ``job_id`` or ``None`` if it is unknown."""
        async with self._lock:
            return self._jobs.get(job_id)

    async def mark_running(self, job_id: str) -> None:
        """Transition a job to ``running``."""
        await self._update(job_id, status="running")

    async def mark_succeeded(self, job_id: str, summary: IngestionSummary) -> None:
        """Record a successful ingestion and attach its summary."""
        await self._update(job_id, status="succeeded", summary=summary)

    async def mark_failed(self, job_id: str, error: str) -> None:
        """Record a failed ingestion and attach a safe error message."""
        await self._update(job_id, status="failed", error=error)

    async def _update(
        self,
        job_id: str,
        *,
        status: JobStatusValue,
        summary: IngestionSummary | None = None,
        error: str | None = None,
    ) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status
            job.updated_at = datetime.now(UTC)
            if summary is not None:
                job.summary = summary
            if error is not None:
                job.error = error
