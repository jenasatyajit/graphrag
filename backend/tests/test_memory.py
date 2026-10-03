"""Conversation store and follow-up rewriter tests."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from src.agent.memory import (
    ConversationTurn,
    FollowUpRewriter,
    InMemoryConversationStore,
    SqliteConversationStore,
)
from src.generation.synthesizer import CompletionResult


def _turn(index: int) -> ConversationTurn:
    return ConversationTurn(
        question=f"q{index}", standalone_question=f"q{index}", answer=f"a{index}"
    )


@pytest.mark.asyncio
async def test_sqlite_store_round_trip_prune_and_isolation(tmp_path: Path) -> None:
    store = SqliteConversationStore(tmp_path / "c.db", max_turns=2, ttl=timedelta(hours=1))
    try:
        a, a_new = await store.resolve(None)
        b, _ = await store.resolve(None)
        for index in range(3):
            await store.append(a, _turn(index))
        await store.append(b, _turn(99))

        assert a_new is True
        assert [t.question for t in await store.history(a, limit=10)] == ["q1", "q2"]
        assert [t.question for t in await store.history(b, limit=10)] == ["q99"]
        assert await store.resolve(a) == (a, False)
        unknown, is_new = await store.resolve("guessed-id")
        assert is_new is True and unknown != "guessed-id"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sqlite_store_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "c.db"
    first = SqliteConversationStore(path, max_turns=5, ttl=timedelta(hours=1))
    conv, _ = await first.resolve(None)
    await first.append(conv, _turn(1))
    await first.close()

    second = SqliteConversationStore(path, max_turns=5, ttl=timedelta(hours=1))
    try:
        assert await second.resolve(conv) == (conv, False)
        assert len(await second.history(conv, limit=5)) == 1
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_expired_conversation_starts_fresh() -> None:
    store = InMemoryConversationStore(max_turns=5, ttl=timedelta(seconds=-1))
    conv, _ = await store.resolve(None)
    await store.append(conv, _turn(1))

    fresh, is_new = await store.resolve(conv)

    assert is_new is True and fresh != conv


class _Rewriter:
    def __init__(self, reply: str | Exception) -> None:
        self._reply = reply
        self.calls = 0

    async def complete(self, *, system_prompt: str, user_prompt: str) -> CompletionResult:
        self.calls += 1
        if isinstance(self._reply, Exception):
            raise self._reply
        return CompletionResult(text=self._reply)


@pytest.mark.asyncio
async def test_rewriter_skips_without_history_and_degrades_on_failure() -> None:
    no_call = _Rewriter("unused")
    assert await FollowUpRewriter(no_call).rewrite("Who approves?", []) == "Who approves?"
    assert no_call.calls == 0

    failing = FollowUpRewriter(_Rewriter(RuntimeError("down")))
    assert await failing.rewrite("and v2?", [_turn(1)]) == "and v2?"

    prefixed = FollowUpRewriter(_Rewriter('Standalone question: "What about SOP v2.1?"'))
    assert await prefixed.rewrite("and v2?", [_turn(1)]) == "What about SOP v2.1?"
