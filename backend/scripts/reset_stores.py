"""Developer utility: wipe the Qdrant collection and the Neo4j graph.

Reads credentials from the same typed ``Settings`` the app uses, so no secrets
are hand-copied. Intended for local Phase 3 testing to get a clean slate before
re-ingesting. Guarded by an explicit ``--yes`` flag because it is destructive.

Usage (from backend/):
    uv run python scripts/reset_stores.py --yes
    uv run python scripts/reset_stores.py --yes --only qdrant
    uv run python scripts/reset_stores.py --yes --only neo4j
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Make ``src`` importable no matter the current working directory: add the
# backend/ root (this file's parent's parent) to sys.path before importing.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import structlog  # noqa: E402

from src.api.config import Settings  # noqa: E402
from src.api.logging import configure_logging  # noqa: E402

logger = structlog.get_logger(__name__)


async def _reset_qdrant(settings: Settings) -> None:
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(
        url=str(settings.qdrant_url),
        api_key=settings.qdrant_api_key.get_secret_value(),
        check_compatibility=False,
    )
    try:
        name = settings.qdrant_collection_name
        exists = await client.collection_exists(name)
        if exists:
            await client.delete_collection(collection_name=name)
            logger.info("qdrant_collection_deleted", collection=name)
        else:
            logger.info("qdrant_collection_absent", collection=name)
    finally:
        await client.close()


async def _reset_neo4j(settings: Settings) -> None:
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
        deleted = result.records[0]["deleted"] if result.records else 0
        logger.info("neo4j_graph_cleared", nodes_deleted=int(deleted))
    finally:
        await driver.close()


async def _main(only: str | None) -> None:
    settings = Settings()
    configure_logging(settings.app_environment)
    if only in (None, "qdrant"):
        await _reset_qdrant(settings)
    if only in (None, "neo4j"):
        await _reset_neo4j(settings)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wipe Qdrant collection and/or Neo4j graph.")
    parser.add_argument("--yes", action="store_true", help="Confirm the destructive reset.")
    parser.add_argument("--only", choices=("qdrant", "neo4j"), help="Reset just one store.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.yes:
        raise SystemExit("Refusing to reset without --yes (this deletes data).")
    asyncio.run(_main(args.only))


if __name__ == "__main__":
    main()
