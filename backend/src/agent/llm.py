"""Tool-calling chat boundary for the router agent (Sarvam-105B).

Verified against ``sarvamai`` 0.1.33 on 2026-09-30: ``chat.completions`` takes
OpenAI-style ``tools`` / ``tool_choice`` (``none|auto|required|named``) and
returns ``message.tool_calls[*].{id, function.name, function.arguments}`` where
``arguments`` is a JSON string. Tool results go back as
``{"role": "tool", "tool_call_id", "content"}`` messages. The router runs with
``reasoning_effort=None`` (reasoning disabled) for latency and cost.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal, Protocol

import structlog
from pydantic import BaseModel, Field
from sarvamai import AsyncSarvamAI
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from src.api.config import Settings
from src.api.errors import LLMError
from src.generation.synthesizer import is_retryable_sarvam_error

logger = structlog.get_logger(__name__)

ToolChoice = Literal["auto", "required", "none"]


class ToolCall(BaseModel):
    """One parsed tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class RouterDecision(BaseModel):
    """The model's reply: tool calls and/or plain text."""

    tool_calls: list[ToolCall] = Field(default_factory=list)
    content: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def assistant_message(self) -> dict[str, Any]:
        """Re-encode this reply as an assistant message for the next turn."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in self.tool_calls
            ]
        return message


class ToolCallingProvider(Protocol):
    async def decide(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]],
        tool_choice: ToolChoice,
    ) -> RouterDecision: ...


class SarvamToolCallingProvider:
    """Resilient Sarvam chat call with tools, reasoning disabled by default."""

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self._model = settings.llm_model
        self._reasoning_effort = settings.llm_router_reasoning_effort
        self._max_tokens = settings.llm_router_max_tokens
        self._max_retries = settings.llm_max_retries
        # Typed as Any: messages/tools are plain OpenAI-shaped dicts, which the
        # SDK's TypedDict overloads do not accept structurally.
        self._client: Any = client or AsyncSarvamAI(
            api_subscription_key=settings.sarvam_api_key.get_secret_value(),
            timeout=settings.llm_request_timeout_seconds,
        )

    async def decide(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]],
        tool_choice: ToolChoice,
    ) -> RouterDecision:
        response: Any = None
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=0.25, min=0.25, max=2),
                retry=retry_if_exception(is_retryable_sarvam_error),
                reraise=True,
            ):
                with attempt:
                    response = await self._client.chat.completions(
                        model=self._model,
                        messages=list(messages),
                        tools=list(tools),
                        tool_choice=tool_choice,
                        reasoning_effort=self._reasoning_effort,
                        max_tokens=self._max_tokens,
                        temperature=0.0,
                    )
        except Exception as exc:
            raise LLMError("The routing model is temporarily unavailable.") from exc
        if response is None or not getattr(response, "choices", None):
            raise LLMError("The routing model returned no choices.")

        message = response.choices[0].message
        calls = [
            call
            for index, raw in enumerate(getattr(message, "tool_calls", None) or [])
            if (call := _parse_tool_call(raw, index)) is not None
        ]
        usage = getattr(response, "usage", None)
        return RouterDecision(
            tool_calls=calls,
            content=getattr(message, "content", None),
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )


def _parse_tool_call(raw: Any, index: int) -> ToolCall | None:
    function = getattr(raw, "function", None)
    name = getattr(function, "name", None)
    if not isinstance(name, str) or not name:
        return None
    arguments = parse_arguments(getattr(function, "arguments", None))
    call_id = getattr(raw, "id", None)
    return ToolCall(
        id=call_id if isinstance(call_id, str) and call_id else f"call_{index}",
        name=name,
        arguments=arguments,
    )


def parse_arguments(raw: Any) -> dict[str, Any]:
    """Decode tool arguments that may arrive as a JSON string or a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}
