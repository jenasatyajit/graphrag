"""Health badge component for the sidebar."""

from __future__ import annotations

import streamlit as st

from api_client import ApiError, HealthResult


def render_health_badge(health: "HealthResult | ApiError") -> None:
    """Render a colored badge plus a per-component caption.

    Green when every component is ``ok``, amber when degraded/503, red when the
    backend is unreachable or misconfigured.
    """
    if isinstance(health, ApiError):
        st.markdown("🔴 **Backend unreachable**")
        st.caption(health.message)
        return

    degraded = health.http_status == 503 or health.status != "ok"
    all_ok = health.status == "ok" and all(c.status == "ok" for c in health.components)

    if all_ok and not degraded:
        st.markdown("🟢 **Online**")
    else:
        st.markdown("🟡 **Degraded**")

    if not health.components:
        st.caption("No component health reported.")
        return

    lines = []
    for c in health.components:
        line = f"- {c.name}: {c.status}"
        if c.detail:
            line += f" — {c.detail}"
        lines.append(line)
    st.caption("\n".join(lines))
