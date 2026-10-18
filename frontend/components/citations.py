"""Rendering helpers for an assistant turn: version warning, tool trace,
notices, and the split Sources panel."""

from __future__ import annotations

import streamlit as st

from api_client import Citation


def render_version_warning(version_warning: str | None) -> None:
    """Show a warning banner when the backend flags a superseded SOP version."""
    if version_warning:
        st.warning(version_warning)


def render_tool_trace(tool_trace: list[str] | None, used_web_search: bool) -> None:
    """Render the router path as a small caption, e.g. ``internal_search → web_search``."""
    if not tool_trace:
        return
    path = " → ".join(str(step) for step in tool_trace)
    suffix = "  ·  web search used" if used_web_search else ""
    st.caption(f"Router path: {path}{suffix}")


def render_notices(notices: list[str] | None) -> None:
    """Show non-claim status notices (e.g. web search unavailable)."""
    for notice in notices or []:
        if notice:
            st.info(notice)


def render_sources(citations: list[Citation] | None) -> None:
    """Expandable Sources panel split into Internal and External sections.

    Renders nothing when there are no citations; the caller decides what (if
    anything) to show in that case. Null-safe on every optional reference field.
    """
    if not citations:
        return

    internal = [c for c in citations if c.source_tier == "internal"]
    external = [c for c in citations if c.source_tier == "external"]

    with st.expander(f"Sources ({len(citations)})", expanded=False):
        if internal:
            st.markdown("**Internal sources**")
            for i, c in enumerate(internal, start=1):
                _render_internal_citation(i, c)
        if external:
            st.markdown("**External sources**")
            for i, c in enumerate(external, start=1):
                _render_external_citation(i, c)


def _render_internal_citation(index: int, citation: Citation) -> None:
    ref = citation.reference
    parts = [f"**{index}. {ref.doc_title}**"]
    if ref.sop_version:
        parts.append(f"version {ref.sop_version}")
    if ref.is_current_version is True:
        parts.append("✅ current")
    elif ref.is_current_version is False:
        parts.append("⚠️ superseded")
    st.markdown("  ·  ".join(parts))
    if citation.claim_text:
        st.caption(f"Supports: {citation.claim_text}")


def _render_external_citation(index: int, citation: Citation) -> None:
    ref = citation.reference
    if ref.url:
        st.markdown(f"**{index}. [{ref.doc_title}]({ref.url})**")
    else:
        st.markdown(f"**{index}. {ref.doc_title}**")
    if ref.retrieved_at:
        st.caption(f"Retrieved: {ref.retrieved_at}")
    if citation.claim_text:
        st.caption(f"Supports: {citation.claim_text}")
