"""Chat view — the core demo. Multi-turn conversation against POST /query."""

from __future__ import annotations

import streamlit as st

from api_client import APIClient, ApiError, QueryResult
from components.citations import (
    render_notices,
    render_sources,
    render_tool_trace,
    render_version_warning,
)


def render(client: APIClient) -> None:
    """Render the chat page: history replay, input, and per-turn layout."""
    st.subheader("Chat")
    st.caption("Ask about company knowledge. Answers include inline [n] citation markers.")

    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("conversation_id", None)

    if st.button("New conversation"):
        st.session_state.chat_messages = []
        st.session_state.conversation_id = None
        st.rerun()

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                _render_assistant_turn(message.get("result"))

    prompt = st.chat_input("Ask about company knowledge…")
    if not prompt:
        return

    st.session_state.chat_messages.append({"role": "user", "content": prompt, "result": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = client.query(prompt, st.session_state.conversation_id)
        _render_assistant_turn(result)

    if isinstance(result, QueryResult) and result.conversation_id:
        # Send the server-issued id back on the next turn to continue the thread.
        st.session_state.conversation_id = result.conversation_id

    st.session_state.chat_messages.append(
        {"role": "assistant", "content": "", "result": result}
    )


def _render_assistant_turn(result: "QueryResult | ApiError | None") -> None:
    """Render one assistant turn in the required order, null-safe throughout."""
    if result is None:
        st.caption("No response recorded.")
        return

    if isinstance(result, ApiError):
        message = result.message
        if result.request_id:
            message += f"\n\nRequest ID: `{result.request_id}`"
        st.error(message)
        return

    # 1. version-warning banner
    render_version_warning(result.version_warning)
    # 2. answer with inline [n] markers
    st.markdown(result.answer or "_(empty answer)_")
    # 3. tool-trace chips/caption
    render_tool_trace(result.tool_trace, result.used_web_search)
    # 4. notices
    render_notices(result.notices)
    # 5. Sources panel (renders nothing when there are no citations)
    render_sources(result.citations)
