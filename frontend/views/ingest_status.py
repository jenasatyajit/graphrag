"""Ingestion status view — renders the last POST /ingest batch summary."""

from __future__ import annotations

import streamlit as st

from api_client import APIClient, ApiError, IngestionSummary


def render(client: APIClient) -> None:
    """Show the latest ingestion batch summary as metrics."""
    st.subheader("Ingestion status")
    st.caption("POST /ingest returns the last batch summary from the GPU host — it does not start a job.")

    if not st.button("Refresh"):
        st.info("Press Refresh to fetch the latest ingestion summary.")
        return

    result = client.ingest_status()
    if isinstance(result, ApiError):
        message = result.message
        if result.request_id:
            message += f"\n\nRequest ID: `{result.request_id}`"
        st.error(message)
        return

    _render_summary(result)


def _render_summary(summary: IngestionSummary) -> None:
    row1 = st.columns(3)
    row1[0].metric("Documents processed", summary.docs_processed)
    row1[1].metric("Chunks created", summary.chunks_created)
    row1[2].metric("Tables extracted", summary.tables_extracted)

    row2 = st.columns(3)
    row2[0].metric("Images extracted", summary.images_extracted)
    row2[1].metric("Graph nodes", summary.graph_nodes_created)
    row2[2].metric("Graph edges", summary.graph_edges_created)

    if summary.completed_at:
        st.caption(f"Completed at: {summary.completed_at}")

    st.markdown("**Failed documents**")
    if summary.failed_documents:
        for name in summary.failed_documents:
            st.markdown(f"- {name}")
    else:
        st.caption("None.")
