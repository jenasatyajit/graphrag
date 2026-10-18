# GraphRAG Frontend

A standalone Streamlit app for the GraphRAG backend. It talks to the public query
API over HTTP via `httpx` — no auth, no token, no CORS concern (requests are made
server-side from the Streamlit process).

Three views, selected from the sidebar:

- **Chat** — multi-turn Q&A against `POST /query`, with inline `[n]` citation
  markers, a router-path caption, version warnings, notices, and a split
  Internal/External sources panel.
- **Ingestion status** — the last `POST /ingest` batch summary as metrics.
- **Reports** — the newest committed evaluation report (falls back to a bundled
  sample).

The sidebar also shows a live backend health badge (green / amber / red).

## Requirements

- Python 3.12 (Streamlit Community Cloud default)
- Two pinned dependencies: `streamlit==1.66.0`, `httpx==0.28.1`

## Run locally

```bash
cd frontend
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit it
streamlit run streamlit_app.py
```

Set `API_BASE_URL` in `.streamlit/secrets.toml` (local dev example
`http://127.0.0.1:8000`). If it is missing, the app shows a setup message instead
of crashing.

> Run from the `frontend/` directory — imports use package paths (`views`,
> `components`).

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Create a new app on [Streamlit Community Cloud](https://share.streamlit.io/)
   pointing at this repo.
3. Set the **main file path** to `frontend/streamlit_app.py`.
4. In the app's **Secrets** dashboard, add (TOML format):

   ```toml
   API_BASE_URL = "https://your-public-api.example.com"
   ```

   The query API must be publicly reachable from the Streamlit Cloud runtime.

## Configuration

| Secret            | Required | Purpose                                                    |
| ----------------- | -------- | ---------------------------------------------------------- |
| `API_BASE_URL`    | yes      | Base URL of the GraphRAG query API.                        |
| `EVAL_REPORT_DIR` | no       | Override the directory the Reports view reads reports from. |

The real `.streamlit/secrets.toml` is git-ignored; only the `.example` is committed.
