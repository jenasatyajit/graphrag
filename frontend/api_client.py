"""Typed HTTP client and client-side models for the GraphRAG backend API.

Models mirror ``backend/src/api/schemas.py`` field-for-field. Every API method
returns either a typed success dataclass or an :class:`ApiError`; exceptions are
never allowed to propagate to the Streamlit UI, so the UI never shows a raw
traceback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

try:  # Streamlit is only imported for secrets access; keep the module importable without it.
    import streamlit as st
except ImportError:  # pragma: no cover - exercised only outside a Streamlit runtime.
    st = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Client-side models (mirror backend/src/api/schemas.py)
# --------------------------------------------------------------------------- #


@dataclass
class CitationReference:
    """Traceable source metadata for a citation (display-only, null-safe)."""

    doc_title: str
    chunk_id: str | None = None
    sop_version: str | None = None
    is_current_version: bool | None = None
    url: str | None = None
    retrieved_at: str | None = None  # ISO 8601 string, shown verbatim.

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CitationReference:
        data = data or {}
        return cls(
            doc_title=data.get("doc_title") or "(untitled source)",
            chunk_id=data.get("chunk_id"),
            sop_version=data.get("sop_version"),
            is_current_version=data.get("is_current_version"),
            url=data.get("url"),
            retrieved_at=data.get("retrieved_at"),
        )


@dataclass
class Citation:
    """A factual claim and the source that supports it."""

    claim_text: str
    source_tier: str  # "internal" | "external"
    reference: CitationReference

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Citation:
        data = data or {}
        return cls(
            claim_text=data.get("claim_text") or "",
            source_tier=data.get("source_tier") or "internal",
            reference=CitationReference.from_dict(data.get("reference")),
        )


@dataclass
class QueryResult:
    """Parsed ``POST /query`` success body."""

    answer: str
    citations: list[Citation]
    used_web_search: bool
    version_warning: str | None
    tool_trace: list[str]
    conversation_id: str
    notices: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueryResult:
        return cls(
            answer=data.get("answer") or "",
            citations=[Citation.from_dict(c) for c in (data.get("citations") or [])],
            used_web_search=bool(data.get("used_web_search", False)),
            version_warning=data.get("version_warning"),
            tool_trace=list(data.get("tool_trace") or []),
            conversation_id=data.get("conversation_id") or "",
            notices=list(data.get("notices") or []),
        )


@dataclass
class ComponentHealth:
    """Health report for a single infrastructure dependency."""

    name: str  # "qdrant" | "neo4j" | "embedding_service"
    status: str  # "ok" | "pending" | "unhealthy"
    detail: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ComponentHealth:
        data = data or {}
        return cls(
            name=data.get("name") or "unknown",
            status=data.get("status") or "unhealthy",
            detail=data.get("detail"),
        )


@dataclass
class HealthResult:
    """Parsed ``GET /health`` body plus transport metadata."""

    status: str  # "ok" | "degraded"
    components: list[ComponentHealth]
    http_status: int
    reachable: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, http_status: int) -> HealthResult:
        return cls(
            status=data.get("status") or "degraded",
            components=[ComponentHealth.from_dict(c) for c in (data.get("components") or [])],
            http_status=http_status,
            reachable=True,
        )


@dataclass
class IngestionSummary:
    """Parsed ``POST /ingest`` body (the last batch summary)."""

    docs_processed: int
    chunks_created: int
    tables_extracted: int
    images_extracted: int
    graph_nodes_created: int
    graph_edges_created: int
    failed_documents: list[str]
    completed_at: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestionSummary:
        def _int(key: str) -> int:
            value = data.get(key)
            try:
                return int(value) if value is not None else 0
            except (TypeError, ValueError):
                return 0

        return cls(
            docs_processed=_int("docs_processed"),
            chunks_created=_int("chunks_created"),
            tables_extracted=_int("tables_extracted"),
            images_extracted=_int("images_extracted"),
            graph_nodes_created=_int("graph_nodes_created"),
            graph_edges_created=_int("graph_edges_created"),
            failed_documents=list(data.get("failed_documents") or []),
            completed_at=data.get("completed_at"),
        )


@dataclass
class JobStarted:
    """Parsed 202 body of ``POST /documents`` (an accepted ingestion job)."""

    job_id: str
    status: str  # always "pending" at creation.

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobStarted:
        return cls(
            job_id=data.get("job_id") or "",
            status=data.get("status") or "pending",
        )


@dataclass
class JobStatus:
    """Parsed ``GET /documents/jobs/{job_id}`` body."""

    job_id: str
    status: str  # "pending" | "running" | "succeeded" | "failed"
    filename: str | None
    doc_type: str | None
    created_at: str | None
    updated_at: str | None
    summary: IngestionSummary | None
    error: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobStatus:
        summary_data = data.get("summary")
        return cls(
            job_id=data.get("job_id") or "",
            status=data.get("status") or "pending",
            filename=data.get("filename"),
            doc_type=data.get("doc_type"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            summary=IngestionSummary.from_dict(summary_data)
            if isinstance(summary_data, dict)
            else None,
            error=data.get("error"),
        )


@dataclass
class ResetStoreOutcome:
    """Result of resetting a single store (mirrors the backend schema)."""

    store: str  # "qdrant" | "neo4j"
    status: str  # "reset" | "skipped"
    detail: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ResetStoreOutcome:
        data = data or {}
        return cls(
            store=data.get("store") or "unknown",
            status=data.get("status") or "skipped",
            detail=data.get("detail"),
        )


@dataclass
class ResetResult:
    """Parsed ``POST /admin/reset`` body describing what was reset."""

    scope: str  # "all" | "qdrant" | "neo4j"
    results: list[ResetStoreOutcome]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResetResult:
        return cls(
            scope=data.get("scope") or "all",
            results=[ResetStoreOutcome.from_dict(r) for r in (data.get("results") or [])],
        )


@dataclass
class ApiError:
    """A friendly, UI-renderable error. Never a raw traceback."""

    message: str
    kind: str  # "config" | "connection" | "timeout" | "http" | "parse"
    request_id: str | None = None
    code: str | None = None
    http_status: int | None = None


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #

_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
# Uploads can be large and the backend reads the whole file before responding,
# so the start-ingestion call gets a more generous read/write budget.
_UPLOAD_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0)


class APIClient:
    """Synchronous client for the GraphRAG backend.

    A sync client suits Streamlit's top-to-bottom rerun model and the
    ``st.spinner`` pattern. Synthesis can take 10-30s, so the read timeout is 60s.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=_TIMEOUT)

    @staticmethod
    def from_secrets() -> "APIClient | ApiError":
        """Build a client from ``st.secrets["API_BASE_URL"]``.

        Returns an ``ApiError(kind="config")`` instead of crashing when the
        secret is missing (Streamlit raises ``StreamlitSecretNotFoundError``).
        """
        config_error = ApiError(
            kind="config",
            message=(
                "API_BASE_URL is not set. Add it in .streamlit/secrets.toml "
                "or the Community Cloud Secrets dashboard."
            ),
        )
        if st is None:
            return config_error
        try:
            base_url = st.secrets["API_BASE_URL"]
        except Exception:  # noqa: BLE001 - StreamlitSecretNotFoundError / KeyError, etc.
            return config_error
        if not base_url or not str(base_url).strip():
            return config_error
        return APIClient(str(base_url).strip())

    # -- public API methods ------------------------------------------------- #

    def health(self) -> "HealthResult | ApiError":
        """GET /health. Both 200 and 503 carry a valid (degraded) body."""
        try:
            resp = self._client.get("/health")
        except httpx.TimeoutException:
            return ApiError(kind="timeout", message="The backend timed out while reporting health.")
        except httpx.RequestError as exc:
            return ApiError(
                kind="connection",
                message=f"Could not reach the backend at {self.base_url}: {exc.__class__.__name__}.",
            )
        if resp.status_code in (200, 503):
            body = self._parse_json(resp)
            if body is None:
                return ApiError(
                    kind="parse",
                    message="The backend returned an unreadable health response.",
                    http_status=resp.status_code,
                )
            return HealthResult.from_dict(body, http_status=resp.status_code)
        return self._error_from_response(resp)

    def query(self, question: str, conversation_id: str | None) -> "QueryResult | ApiError":
        """POST /query. First turn sends conversation_id=None; then send server value back."""
        payload = {"question": question, "conversation_id": conversation_id}
        try:
            resp = self._client.post("/query", json=payload)
        except httpx.TimeoutException:
            return ApiError(
                kind="timeout",
                message="The backend took too long to answer (over 60s). Try a simpler question.",
            )
        except httpx.RequestError as exc:
            return ApiError(
                kind="connection",
                message=f"Could not reach the backend at {self.base_url}: {exc.__class__.__name__}.",
            )
        if resp.status_code == 200:
            body = self._parse_json(resp)
            if body is None:
                return ApiError(kind="parse", message="The backend returned an unreadable answer.")
            return QueryResult.from_dict(body)
        return self._error_from_response(resp)

    def ingest_status(self) -> "IngestionSummary | ApiError":
        """POST /ingest (no body). Returns the last batch summary; starts no job."""
        try:
            resp = self._client.post("/ingest")
        except httpx.TimeoutException:
            return ApiError(kind="timeout", message="The backend timed out while reporting ingestion status.")
        except httpx.RequestError as exc:
            return ApiError(
                kind="connection",
                message=f"Could not reach the backend at {self.base_url}: {exc.__class__.__name__}.",
            )
        if resp.status_code == 200:
            body = self._parse_json(resp)
            if body is None:
                return ApiError(kind="parse", message="The backend returned an unreadable ingestion summary.")
            return IngestionSummary.from_dict(body)
        return self._error_from_response(resp)

    def start_ingestion(
        self,
        file_bytes: bytes,
        filename: str,
        doc_type: str,
        sop_version: str | None = None,
        doc_title: str | None = None,
        effective_date: str | None = None,
    ) -> "JobStarted | ApiError":
        """POST /documents (multipart). Returns the accepted job or an ApiError.

        A 409 response (an ingestion is already running) is surfaced as an
        ``ApiError(kind="http", http_status=409)`` with the backend message.
        """
        data: dict[str, str] = {"doc_type": doc_type}
        if sop_version:
            data["sop_version"] = sop_version
        if doc_title:
            data["doc_title"] = doc_title
        if effective_date:
            data["effective_date"] = effective_date
        files = {"file": (filename, file_bytes)}
        try:
            resp = self._client.post(
                "/documents", files=files, data=data, timeout=_UPLOAD_TIMEOUT
            )
        except httpx.TimeoutException:
            return ApiError(
                kind="timeout", message="The upload timed out while contacting the backend."
            )
        except httpx.RequestError as exc:
            return ApiError(
                kind="connection",
                message=f"Could not reach the backend at {self.base_url}: {exc.__class__.__name__}.",
            )
        if resp.status_code == 202:
            body = self._parse_json(resp)
            if body is None:
                return ApiError(kind="parse", message="The backend returned an unreadable response.")
            return JobStarted.from_dict(body)
        return self._error_from_response(resp)

    def ingestion_job(self, job_id: str) -> "JobStatus | ApiError":
        """GET /documents/jobs/{job_id}. Returns the job status or an ApiError."""
        try:
            resp = self._client.get(f"/documents/jobs/{job_id}")
        except httpx.TimeoutException:
            return ApiError(
                kind="timeout", message="The backend timed out while reporting job status."
            )
        except httpx.RequestError as exc:
            return ApiError(
                kind="connection",
                message=f"Could not reach the backend at {self.base_url}: {exc.__class__.__name__}.",
            )
        if resp.status_code == 200:
            body = self._parse_json(resp)
            if body is None:
                return ApiError(kind="parse", message="The backend returned an unreadable job status.")
            return JobStatus.from_dict(body)
        return self._error_from_response(resp)

    def reset_stores(self, scope: str, confirm: bool) -> "ResetResult | ApiError":
        """POST /admin/reset. Returns the per-store outcome or an ApiError."""
        payload = {"scope": scope, "confirm": confirm}
        try:
            resp = self._client.post("/admin/reset", json=payload)
        except httpx.TimeoutException:
            return ApiError(kind="timeout", message="The backend timed out while resetting stores.")
        except httpx.RequestError as exc:
            return ApiError(
                kind="connection",
                message=f"Could not reach the backend at {self.base_url}: {exc.__class__.__name__}.",
            )
        if resp.status_code == 200:
            body = self._parse_json(resp)
            if body is None:
                return ApiError(kind="parse", message="The backend returned an unreadable reset result.")
            return ResetResult.from_dict(body)
        return self._error_from_response(resp)

    # -- helpers ------------------------------------------------------------ #

    @staticmethod
    def _parse_json(resp: httpx.Response) -> dict[str, Any] | None:
        try:
            data = resp.json()
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def _error_from_response(self, resp: httpx.Response) -> ApiError:
        """Parse the ``{"error": {...}}`` envelope; fall back to raw text."""
        body = self._parse_json(resp)
        if body and isinstance(body.get("error"), dict):
            err = body["error"]
            return ApiError(
                kind="http",
                message=err.get("message") or "The backend returned an error.",
                code=err.get("code"),
                request_id=err.get("request_id"),
                http_status=resp.status_code,
            )
        snippet = (resp.text or "").strip()[:300] or "The backend returned an error."
        return ApiError(kind="http", message=snippet, http_status=resp.status_code)
