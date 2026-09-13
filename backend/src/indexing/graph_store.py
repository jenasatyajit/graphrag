"""Neo4j client wrapper placeholder for Phase 3."""

from __future__ import annotations

from src.api.config import Settings
from src.api.schemas import ComponentHealth


class GraphStore:
    """Thin boundary around the Neo4j client.

    TODO(phase-3): create the Neo4j driver and implement a real connectivity
    check before graph persistence and traversal are introduced.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def connect(self) -> None:
        """Reserve an explicit connection boundary for the Phase 3 Neo4j driver."""
        return None

    async def verify_connectivity(self) -> ComponentHealth:
        """Report that the Phase 3 Neo4j connectivity check is pending."""
        return ComponentHealth(
            name="neo4j",
            status="pending",
            detail="TODO(phase-3): Neo4j connectivity verification is not implemented.",
        )
