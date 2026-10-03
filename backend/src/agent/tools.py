"""Tool schemas exposed to the router agent (SPEC 7.4).

The schemas use the OpenAI-compatible ``{"type": "function", ...}`` shape that
Sarvam chat completions accept (only function tools are supported).
"""

from __future__ import annotations

from typing import Any, Final, Literal

ToolName = Literal["internal_search", "web_search", "answer_directly"]

INTERNAL_SEARCH: Final = "internal_search"
WEB_SEARCH: Final = "web_search"
ANSWER_DIRECTLY: Final = "answer_directly"
TOOL_NAMES: Final[frozenset[str]] = frozenset({INTERNAL_SEARCH, WEB_SEARCH, ANSWER_DIRECTLY})

_QUERY_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "A standalone search query. Rephrase it if an earlier search failed.",
        }
    },
    "required": ["query"],
}

ROUTER_TOOLS: Final[list[dict[str, Any]]] = [
    {
        "type": "function",
        "function": {
            "name": INTERNAL_SEARCH,
            "description": (
                "Search internal SOPs and product docs (hybrid vector search plus the "
                "knowledge graph). Use for anything about company processes, roles, "
                "systems, compliance, SOP versions, or products."
            ),
            "parameters": _QUERY_PARAMETERS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": WEB_SEARCH,
            "description": (
                "Search the live web for current external information such as public "
                "regulations, market data, news, or third-party/competitor facts. Results "
                "are labelled external, never company policy."
            ),
            "parameters": _QUERY_PARAMETERS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": ANSWER_DIRECTLY,
            "description": (
                "Use only if the message is a greeting, thanks, or a meta-question about "
                "this assistant that needs no retrieval."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why no retrieval is needed.",
                    }
                },
                "required": [],
            },
        },
    },
]
