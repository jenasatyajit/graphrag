"""Phase 7 logging audit: structured logs carry ids/counts, not document bodies.

SPEC 12 and BACKEND.md 4.5 require tool calls, retrieval counts, rerank scores,
and token usage to be logged, but forbid logging secrets, PII, or full document
bodies. The router's per-tool-call summary (fed to the LLM) contains chunk
excerpts and web snippets; this test proves the structured ``router_tool_call``
log event has those text-bearing fields redacted while keeping the useful
counts/flags.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import structlog

from src.agent.router import _redacted_summary
from src.generation.synthesizer import AnswerSynthesizer
from tests.conftest import ScriptedRouterLLM, tool_decision
from tests.test_router import ScriptedCompleter, StubInternal, _chunk, _router


@pytest.fixture(autouse=True)
def _plain_logging() -> Iterator[None]:
    """Render logs to stdout with a plain key=value renderer for assertions.

    The module-level loggers in src are resolved and cached at import time (the
    production config sets cache_logger_on_first_use=True), so capture_logs'
    processor swap cannot intercept them. Reconfiguring the shared pipeline here
    makes every log event render to stdout as deterministic key=value pairs that
    the test can scan for leaked document bodies. Defaults are restored after.
    """
    structlog.configure(
        processors=[structlog.processors.KeyValueRenderer(sort_keys=True)],
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    yield
    structlog.reset_defaults()


def test_redacted_summary_drops_text_fields_keeps_counts() -> None:
    summary = {
        "tool": "internal_search",
        "relevant": True,
        "result_count": 2,
        "top_rerank_score": 0.91,
        "results": [{"excerpt": "SECRET BODY TEXT", "title": "SOP"}],
        "graph_facts": ["Expense Approval requires SAP"],
    }

    redacted = _redacted_summary(summary)

    assert redacted == {
        "tool": "internal_search",
        "relevant": True,
        "result_count": 2,
        "top_rerank_score": 0.91,
    }
    assert "results" not in redacted
    assert "graph_facts" not in redacted


@pytest.mark.asyncio
async def test_router_tool_call_log_omits_chunk_text(capsys: pytest.CaptureFixture[str]) -> None:
    # The _plain_logging fixture renders every log event to stdout as key=value
    # pairs; scanning that output proves the retrieved body never reaches a log,
    # independent of which logger instance src cached at import time.
    secret_text = "CONFIDENTIAL-BODY-DO-NOT-LOG expenses above $500."
    router = _router(
        llm=ScriptedRouterLLM(tool_decision("internal_search")),
        internal=StubInternal([_chunk(secret_text)]),
        synth=ScriptedCompleter("Managers approve [1]."),
    )

    await router.answer("Who approves expenses?")

    out = capsys.readouterr().out
    tool_lines = [line for line in out.splitlines() if "router_tool_call" in line]
    assert tool_lines, "expected a router_tool_call log line"
    # The retrieved document body is never written to any log line.
    assert secret_text not in out
    assert "excerpt" not in out
    # The useful signal (counts) is still logged on the tool-call event.
    assert any("result_count" in line for line in tool_lines)


@pytest.mark.asyncio
async def test_answer_synthesized_log_reports_counts_not_bodies(
    capsys: pytest.CaptureFixture[str],
) -> None:
    synth = AnswerSynthesizer(ScriptedCompleter("Managers approve expenses [1]."))

    await synth.synthesize("Who approves expenses?", [_chunk().model_copy()])

    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if "answer_synthesized" in line]
    assert lines
    line = lines[0]
    assert "source_count" in line and "citation_count" in line
    # Token-usage keys are present (SPEC 7.4/12), even if None under the fake.
    assert "input_tokens" in line and "output_tokens" in line
