"""GraphRAG Streamlit frontend — entry point.

Run from this directory:  ``streamlit run streamlit_app.py``
Imports use package paths (``views``, ``components``), so the working directory
must be ``frontend/``.
"""

from __future__ import annotations

import streamlit as st

from api_client import APIClient, ApiError
from components.status import render_health_badge
from views import chat, ingest_status, reports

st.set_page_config(page_title="GraphRAG", layout="wide")

_PAGES = ["Chat", "Ingestion status", "Reports"]


def _sidebar(client: "APIClient | ApiError") -> str:
    st.sidebar.title("GraphRAG")
    page = st.sidebar.radio("View", _PAGES)

    st.sidebar.divider()
    st.sidebar.markdown("**Backend**")
    if isinstance(client, ApiError):
        render_health_badge(client)
    else:
        render_health_badge(client.health())
    return page


def main() -> None:
    client = APIClient.from_secrets()
    page = _sidebar(client)

    if page == "Reports":
        # Reports read a local file and do not need the backend client.
        reports.render()
        return

    if isinstance(client, ApiError):
        st.subheader(page)
        st.warning(
            "Configure the backend first: set `API_BASE_URL` in "
            "`.streamlit/secrets.toml` (local dev example `http://127.0.0.1:8000`) "
            "or in the Streamlit Community Cloud Secrets dashboard."
        )
        return

    if page == "Chat":
        chat.render(client)
    elif page == "Ingestion status":
        ingest_status.render(client)


main()
