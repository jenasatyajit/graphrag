"""Single source of truth for wiping the Qdrant collection and Neo4j graph.

Both the developer CLI (``scripts/reset_stores.py``) and the ``POST /admin/reset``
endpoint call these helpers so there is exactly one implementation. The Qdrant
and Neo4j client libraries are imported lazily inside each function to keep this
module CPU-safe and cheap to import; every external client is closed in a
``finally`` block so a failure mid-reset never leaks a connection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import structlog

from src.api.schemas import ResetStoreOutcome

if TYPE_CHECKING:
    from src.api.config import Settings

logger = structlog.get_logger(__name__)

ResetScope = Literal["all", "qdrant", "neo4j"]


async def reset_qdrant(settings: Settings) -> ResetStoreOutcome:
    """Delete the configured Qdrant collection; skip quietly if it is absent."""
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(
        url=str(settings.qdrant_url),
        api_key=settings.qdrant_api_key.get_secret_value(),
        check_compatibility=False,
    )
    name = settings.qdrant_collection_name
    try:
        if await client.collection_exists(name):
            await client.delete_collection(collection_name=name)
            logger.info("qdrant_collection_deleted", collection=name)
            return ResetStoreOutcome(
                store="qdrant",
                status="reset",
                detail=f"Deleted collection '{name}'.",
            )
        logger.info("qdrant_collection_absent", collection=name)
        return ResetStoreOutcome(
            store="qdrant",
            status="skipped",
            detail=f"Collection '{name}' did not exist.",
        )
    finally:
        await client.close()


async def reset_neo4j(settings: Settings) -> ResetStoreOutcome:
    """Detach-delete every node in the configured Neo4j database."""
    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
    )
    try:
        await driver.verify_connectivity()
        result = await driver.execute_query(
            "MATCH (n) DETACH DELETE n RETURN count(n) AS deleted",
            database_=settings.neo4j_database,
            routing_="w",
        )
        deleted = int(result.records[0]["deleted"]) if result.records else 0
        logger.info("neo4j_graph_cleared", nodes_deleted=deleted)
        return ResetStoreOutcome(
            store="neo4j",
            status="reset",
            detail=f"Deleted {deleted} node(s).",
        )
    finally:
        await driver.close()


async def reset_stores(settings: Settings, *, scope: ResetScope = "all") -> list[ResetStoreOutcome]:
    """Reset the stores selected by ``scope`` and return a per-store outcome list."""
    results: list[ResetStoreOutcome] = []
    if scope in ("all", "qdrant"):
        results.append(await reset_qdrant(settings))
    if scope in ("all", "neo4j"):
        results.append(await reset_neo4j(settings))
    return results
