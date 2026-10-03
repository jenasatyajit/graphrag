"""Multi-turn conversation memory (BACKEND.md 4.9, deviation D-10).

History is scoped strictly by ``conversation_id`` and lives only in the
conversation store: it is never written to Qdrant or Neo4j and is never a
citation source. Ids are issued by the server (random UUID4); an unknown or
expired client-supplied id starts a fresh conversation under a new id, so a
caller cannot read another conversation by guessing or reusing an id that the
server has not issued to it recently.
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import structlog
from pydantic import BaseModel, Field

from src.agent.prompts import FOLLOW_UP_REWRITE_SYSTEM_PROMPT
from src.api.config import Settings
from src.generation.synthesizer import CompletionProvider

logger = structlog.get_logger(__name__)

# Prior answers are truncated when shown to the rewriter/synthesizer so history
# cannot crowd out retrieved evidence or balloon token cost.
_ANSWER_PREVIEW_CHARS = 600


class ConversationTurn(BaseModel):
    """One completed question/answer exchange."""

    question: str
    standalone_question: str
    answer: str
    tool_trace: list[str] = Field(default_factory=list)
    used_web_search: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConversationStore(Protocol):
    async def resolve(self, conversation_id: str | None) -> tuple[str, bool]:
        """Return ``(id, is_new)``; unknown/expired ids yield a fresh id."""
        ...

    async def history(self, conversation_id: str, *, limit: int) -> list[ConversationTurn]: ...

    async def append(self, conversation_id: str, turn: ConversationTurn) -> None: ...

    async def close(self) -> None: ...


def _new_id() -> str:
    return str(uuid.uuid4())


class InMemoryConversationStore:
    """Process-local store for development and tests."""

    def __init__(self, *, max_turns: int, ttl: timedelta) -> None:
        self._max_turns = max_turns
        self._ttl = ttl
        self._turns: dict[str, list[ConversationTurn]] = {}
        self._touched: dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def resolve(self, conversation_id: str | None) -> tuple[str, bool]:
        async with self._lock:
            self._evict_expired()
            if conversation_id and conversation_id in self._turns:
                return conversation_id, False
            new_id = _new_id()
            self._turns[new_id] = []
            self._touched[new_id] = datetime.now(UTC)
            return new_id, True

    async def history(self, conversation_id: str, *, limit: int) -> list[ConversationTurn]:
        async with self._lock:
            turns = self._turns.get(conversation_id, [])
            return [turn.model_copy() for turn in turns[-limit:]]

    async def append(self, conversation_id: str, turn: ConversationTurn) -> None:
        async with self._lock:
            turns = self._turns.setdefault(conversation_id, [])
            turns.append(turn)
            del turns[: max(0, len(turns) - self._max_turns)]
            self._touched[conversation_id] = datetime.now(UTC)

    async def close(self) -> None:
        return None

    def _evict_expired(self) -> None:
        cutoff = datetime.now(UTC) - self._ttl
        for key in [key for key, touched in self._touched.items() if touched < cutoff]:
            self._turns.pop(key, None)
            self._touched.pop(key, None)


class SqliteConversationStore:
    """Lightweight persistent store (stdlib ``sqlite3`` run off the event loop)."""

    def __init__(self, path: Path, *, max_turns: int, ttl: timedelta) -> None:
        self._path = path
        self._max_turns = max_turns
        self._ttl = ttl
        self._lock = asyncio.Lock()
        self._conn: sqlite3.Connection | None = None

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL
                        REFERENCES conversations(id) ON DELETE CASCADE,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS turns_by_conversation
                    ON turns(conversation_id, seq);
                """
            )
            conn.commit()
            self._conn = conn
        return self._conn

    async def _run[T](self, fn: Callable[[sqlite3.Connection], T]) -> T:
        async with self._lock:
            return await asyncio.to_thread(fn, self._connection())

    async def resolve(self, conversation_id: str | None) -> tuple[str, bool]:
        now = datetime.now(UTC)
        cutoff = (now - self._ttl).isoformat()

        def _resolve(conn: sqlite3.Connection) -> tuple[str, bool]:
            conn.execute("DELETE FROM conversations WHERE updated_at < ?", (cutoff,))
            if conversation_id:
                row = conn.execute(
                    "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()
                if row is not None:
                    conn.commit()
                    return conversation_id, False
            new_id = _new_id()
            conn.execute(
                "INSERT INTO conversations (id, updated_at) VALUES (?, ?)",
                (new_id, now.isoformat()),
            )
            conn.commit()
            return new_id, True

        return await self._run(_resolve)

    async def history(self, conversation_id: str, *, limit: int) -> list[ConversationTurn]:
        def _history(conn: sqlite3.Connection) -> list[ConversationTurn]:
            rows = conn.execute(
                "SELECT payload FROM turns WHERE conversation_id = ? ORDER BY seq DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
            return [ConversationTurn.model_validate_json(row[0]) for row in reversed(rows)]

        return await self._run(_history)

    async def append(self, conversation_id: str, turn: ConversationTurn) -> None:
        now = datetime.now(UTC).isoformat()

        def _append(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO conversations (id, updated_at) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at",
                (conversation_id, now),
            )
            conn.execute(
                "INSERT INTO turns (conversation_id, payload) VALUES (?, ?)",
                (conversation_id, turn.model_dump_json()),
            )
            conn.execute(
                "DELETE FROM turns WHERE conversation_id = ? AND seq NOT IN ("
                "SELECT seq FROM turns WHERE conversation_id = ? ORDER BY seq DESC LIMIT ?)",
                (conversation_id, conversation_id, self._max_turns),
            )
            conn.commit()

        await self._run(_append)

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                await asyncio.to_thread(self._conn.close)
                self._conn = None


def build_conversation_store(settings: Settings) -> ConversationStore:
    """Select the configured conversation backend."""
    ttl = timedelta(hours=settings.conversation_ttl_hours)
    if settings.conversation_store == "sqlite":
        return SqliteConversationStore(
            settings.conversation_db_path,
            max_turns=settings.conversation_history_max_turns,
            ttl=ttl,
        )
    return InMemoryConversationStore(
        max_turns=settings.conversation_history_max_turns,
        ttl=ttl,
    )


def render_history(turns: Sequence[ConversationTurn]) -> str:
    """Render prior turns as plain background text (never numbered/citable)."""
    lines: list[str] = []
    for turn in turns:
        answer = turn.answer.strip()
        if len(answer) > _ANSWER_PREVIEW_CHARS:
            answer = answer[:_ANSWER_PREVIEW_CHARS].rstrip() + " ..."
        lines.append(f"User: {turn.standalone_question or turn.question}")
        lines.append(f"Assistant: {answer}")
    return "\n".join(lines)


class FollowUpRewriter:
    """Rewrite a terse follow-up into a standalone question using history."""

    def __init__(self, completion_provider: CompletionProvider) -> None:
        self._provider = completion_provider

    async def rewrite(self, question: str, history: Sequence[ConversationTurn]) -> str:
        if not history:
            return question
        try:
            completion = await self._provider.complete(
                system_prompt=FOLLOW_UP_REWRITE_SYSTEM_PROMPT,
                user_prompt=(
                    f"Conversation so far:\n{render_history(history)}\n\n"
                    f"Latest message:\n{question.strip()}"
                ),
            )
            rewritten = _clean_rewrite(completion.text)
        except Exception:
            logger.warning("follow_up_rewrite_failed", exc_info=True)
            return question
        if not rewritten:
            return question
        logger.info(
            "follow_up_rewritten",
            history_turns=len(history),
            changed=rewritten != question.strip(),
        )
        return rewritten


def _clean_rewrite(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    first = lines[0]
    for prefix in ("Rewritten question:", "Standalone question:", "Question:"):
        if first.lower().startswith(prefix.lower()):
            first = first[len(prefix) :].strip()
    return first.strip("\"'` ").strip()
