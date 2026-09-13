"""Qdrant client wrapper placeholder for Phase 1."""

from __future__ import annotations

from src.api.config import Settings
from src.api.schemas import ComponentHealth


class VectorStore:
    """Thin boundary around the Qdrant client.

    TODO(phase-1): instantiate the Qdrant client and perform a real connection
    check before supporting collection creation, upserts, and search.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def connect(self) -> None:
        """Reserve an explicit connection boundary for the Phase 1 Qdrant client."""
        return None

    async def verify_connectivity(self) -> ComponentHealth:
        """Report that the Phase 1 Qdrant connectivity check is pending."""
        return ComponentHealth(
            name="qdrant",
            status="pending",
            detail="TODO(phase-1): Qdrant connectivity verification is not implemented.",
        )
