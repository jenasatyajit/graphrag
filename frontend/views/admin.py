"""Admin view — ingest a document and reset the data stores.

Both actions hit the intentionally unauthenticated admin endpoints. Ingestion
runs asynchronously on the backend, so this view starts a job, tracks the job id
in ``st.session_state`` across reruns, and polls the status endpoint on a timed
rerun loop until the job reaches a terminal state.
"""

from __future__ import annotations

import time
from datetime import date

import streamlit as st

from api_client import APIClient, ApiError, IngestionSummary, JobStarted, JobStatus, ResetResult

_POLL_INTERVAL_SECONDS = 2.0
_TERMINAL_STATUSES = frozenset({"succeeded", "failed"})
_SCOPE_LABELS = {
    "All stores (Qdrant + Neo4j)": "all",
    "Qdrant only": "qdrant",
    "Neo4j only": "neo4j",
}


def render(client: APIClient) -> None:
    """Render the Admin page with ingest and reset sections."""
    st.subheader("Admin")
    st.caption("Manage ingested data. These actions are unauthenticated in this demo.")

    _render_ingest(client)
    st.divider()
    _render_reset(client)


def _render_ingest(client: APIClient) -> None:
    st.markdown("### Ingest a document")
    st.caption("Upload one file and start a background ingestion job on the GPU host.")

    st.session_state.setdefault("admin_job_id", None)
    active_job_id = st.session_state.admin_job_id

    uploaded = st.file_uploader("Document", type=["pdf", "docx", "md"])
    doc_type = st.selectbox("Document type", ["SOP", "product_info"])

    sop_version: str | None = None
    doc_title: str | None = None
    effective_date_value: date | None = None
    if doc_type == "SOP":
        sop_version = st.text_input("SOP version (optional)") or None
        doc_title = st.text_input("Document title (optional)") or None
        use_effective_date = st.checkbox("Set an effective date")
        if use_effective_date:
            effective_date_value = st.date_input("Effective date")

    ingest_disabled = active_job_id is not None
    if st.button("Ingest", disabled=ingest_disabled):
        if uploaded is None:
            st.warning("Choose a file to ingest first.")
        else:
            effective_date_iso = (
                effective_date_value.isoformat() if effective_date_value is not None else None
            )
            result = client.start_ingestion(
                file_bytes=uploaded.getvalue(),
                filename=uploaded.name,
                doc_type=doc_type,
                sop_version=sop_version,
                doc_title=doc_title,
                effective_date=effective_date_iso,
            )
            if isinstance(result, ApiError):
                _render_start_error(result)
            elif isinstance(result, JobStarted):
                st.session_state.admin_job_id = result.job_id
                st.rerun()

    if st.session_state.admin_job_id is not None:
        _poll_job(client, st.session_state.admin_job_id)


def _render_start_error(error: ApiError) -> None:
    if error.http_status == 409:
        st.warning("An ingestion is already running. Wait for it to finish, then try again.")
        return
    message = error.message
    if error.request_id:
        message += f"\n\nRequest ID: `{error.request_id}`"
    st.error(message)


def _poll_job(client: APIClient, job_id: str) -> None:
    status = client.ingestion_job(job_id)
    if isinstance(status, ApiError):
        st.error(status.message)
        st.session_state.admin_job_id = None
        return

    if status.status in _TERMINAL_STATUSES:
        _render_terminal(status)
        st.session_state.admin_job_id = None
        return

    st.info(f"Ingestion job `{job_id}` is **{status.status}**…")
    with st.spinner("Ingesting… this can take a while on large documents."):
        time.sleep(_POLL_INTERVAL_SECONDS)
    st.rerun()


def _render_terminal(status: JobStatus) -> None:
    if status.status == "succeeded" and status.summary is not None:
        st.success(f"Ingestion job `{status.job_id}` succeeded.")
        _render_summary(status.summary)
    elif status.status == "failed":
        st.error(f"Ingestion job `{status.job_id}` failed: {status.error or 'unknown error'}")
    else:
        st.warning(f"Ingestion job `{status.job_id}` finished with no summary.")


def _render_summary(summary: IngestionSummary) -> None:
    row1 = st.columns(3)
    row1[0].metric("Documents processed", summary.docs_processed)
    row1[1].metric("Chunks created", summary.chunks_created)
    row1[2].metric("Tables extracted", summary.tables_extracted)

    row2 = st.columns(3)
    row2[0].metric("Images extracted", summary.images_extracted)
    row2[1].metric("Graph nodes", summary.graph_nodes_created)
    row2[2].metric("Graph edges", summary.graph_edges_created)

    if summary.failed_documents:
        st.markdown("**Failed documents**")
        for name in summary.failed_documents:
            st.markdown(f"- {name}")


def _render_reset(client: APIClient) -> None:
    st.markdown("### Reset data stores")
    st.warning(
        "This permanently deletes ingested data from the selected stores. "
        "It cannot be undone."
    )

    scope_label = st.radio("Scope", list(_SCOPE_LABELS.keys()))
    scope = _SCOPE_LABELS[scope_label]

    acknowledged = st.checkbox("I understand this permanently deletes ingested data")
    if st.button("Reset", type="primary", disabled=not acknowledged):
        result = client.reset_stores(scope=scope, confirm=True)
        if isinstance(result, ApiError):
            message = result.message
            if result.request_id:
                message += f"\n\nRequest ID: `{result.request_id}`"
            st.error(message)
            return
        _render_reset_result(result)


def _render_reset_result(result: ResetResult) -> None:
    st.success(f"Reset complete (scope: {result.scope}).")
    for outcome in result.results:
        line = f"**{outcome.store}**: {outcome.status}"
        if outcome.detail:
            line += f" — {outcome.detail}"
        st.markdown(f"- {line}")
