"""Background ingestion task that crosses the CPU/GPU boundary lazily.

The single GPU-touching import (``scripts.run_ingestion.run_ingestion``, which
pulls in Docling and BGE-M3) is performed INSIDE the task body, never at module
import time, so importing this module stays CPU-safe. If the GPU dependency
group is not installed, the import fails and the JOB is marked failed with a
clear message — the application never crashes.
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.api.config import Settings
    from src.api.jobs import JobStore

logger = structlog.get_logger(__name__)

_GPU_MISSING_MESSAGE = "ingestion requires the GPU host with the gpu dependency group installed"


async def run_ingestion_job(
    job_store: JobStore,
    settings: Settings,
    *,
    job_id: str,
    job_dir: Path,
    doc_type: str,
    sop_version: str | None,
    doc_title: str | None,
    effective_date: date | None,
) -> None:
    """Run one ingestion job over ``job_dir`` and record its outcome.

    Imports ``run_ingestion`` lazily, points it at the per-job input directory
    via a settings override, and always deletes the per-job temp directory when
    finished. All failures are captured on the job record with a safe message.
    """
    await job_store.mark_running(job_id)
    try:
        from typing import cast

        from scripts.run_ingestion import run_ingestion
        from src.ingestion.parser import DocType

        job_settings = settings.model_copy(update={"ingestion_data_dir": job_dir})
        summary = await run_ingestion(
            job_settings,
            doc_type=cast(DocType, doc_type),
            sop_version=sop_version,
            effective_date=effective_date,
            doc_title=doc_title,
        )
        await job_store.mark_succeeded(job_id, summary)
        logger.info("ingestion_job_succeeded", job_id=job_id)
    except (ImportError, ModuleNotFoundError):
        logger.warning("ingestion_job_gpu_unavailable", job_id=job_id, exc_info=True)
        await job_store.mark_failed(job_id, _GPU_MISSING_MESSAGE)
    except Exception:
        logger.exception("ingestion_job_failed", job_id=job_id)
        await job_store.mark_failed(job_id, "Ingestion failed. See server logs for details.")
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
