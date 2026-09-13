"""Structured logging configuration for the FastAPI service."""

from __future__ import annotations

import logging
import sys
from typing import Literal

import structlog


def configure_logging(environment: Literal["development", "production", "test"]) -> None:
    """Configure stdlib and structlog output once for the running process."""
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout, force=True)

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if environment == "production":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
