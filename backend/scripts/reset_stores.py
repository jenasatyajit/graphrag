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

from src.admin.store_reset import reset_neo4j, reset_qdrant  # noqa: E402
from src.api.config import Settings  # noqa: E402
from src.api.logging import configure_logging  # noqa: E402

logger = structlog.get_logger(__name__)


async def _main(only: str | None) -> None:
    settings = Settings()
    configure_logging(settings.app_environment)
    if only in (None, "qdrant"):
        await reset_qdrant(settings)
    if only in (None, "neo4j"):
        await reset_neo4j(settings)


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
