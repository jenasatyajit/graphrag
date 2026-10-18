"""Reports view — render the newest committed evaluation report.

Reads the newest ``*.md`` from the backend eval reports directory, falling back
to the bundled ``sample_report.md``. No backend endpoint is involved (deferred).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_FRONTEND_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_REPORTS_DIR = _FRONTEND_DIR.parent / "backend" / "data" / "eval" / "reports"
_SAMPLE_REPORT = _FRONTEND_DIR / "sample_report.md"


def _reports_dir() -> Path:
    """Resolve the reports directory, configurable via EVAL_REPORT_DIR secret."""
    try:
        configured = st.secrets.get("EVAL_REPORT_DIR")
    except Exception:  # noqa: BLE001 - secrets may be entirely unset.
        configured = None
    if configured:
        return Path(str(configured))
    return _DEFAULT_REPORTS_DIR


def _newest_report(directory: Path) -> Path | None:
    try:
        if not directory.is_dir():
            return None
        candidates = sorted(
            directory.glob("*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None
    return candidates[0] if candidates else None


def render() -> None:
    """Render the newest committed report, else the bundled sample."""
    st.subheader("Evaluation report")

    report_path = _newest_report(_reports_dir())
    is_sample = False

    if report_path is None:
        report_path = _SAMPLE_REPORT if _SAMPLE_REPORT.is_file() else None
        is_sample = True

    if report_path is None:
        st.info(
            "No evaluation report available yet — run the eval harness and commit the report."
        )
        return

    try:
        text = report_path.read_text(encoding="utf-8")
    except OSError as exc:
        st.error(f"Could not read the report file: {exc.__class__.__name__}.")
        return

    if is_sample:
        st.info("Showing bundled sample data — no committed evaluation report found.")
    st.caption(f"Source: {report_path.name}")
    st.markdown(text)
