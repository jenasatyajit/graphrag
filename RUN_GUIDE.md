# Run Guide — GraphRAG Company-Knowledge QA

End-to-end instructions to run the whole application: ingestion → query API →
frontend → evaluation. For *what* the system is, see
[`docs/SPEC.md`](docs/SPEC.md); for the build plan, [`docs/BACKEND.md`](docs/BACKEND.md);
for the operational runbook, [`docs/RUNNING.md`](docs/RUNNING.md).

---

## 1. Topology at a glance

The system runs across two machines plus cloud services. The split is deliberate:
BGE-M3 embeddings and Docling's vision pipeline want a GPU; the query API does not.

```
GPU VM (Ubuntu + NVIDIA L40S)              Cloud services         Frontend
──────────────────────────────            ──────────────         ──────────────────
embedding_service   :8081                 Qdrant Cloud           Streamlit app
run_ingestion.py (batch, Docling+BGE-M3)  Neo4j AuraDB           (Community Cloud
query API           :8002 /8000           Sarvam / OpenRouter     or local)
(optional) vLLM VLM :8000                 / Exa                  → talks to query API
```

- **Ingestion** runs only on the GPU VM (needs the GPU). It writes chunks to
  Qdrant Cloud and the graph to Neo4j AuraDB.
- **The query API** can run on the GPU VM (simplest for a demo) or a separate
  CPU box that reaches the VM's embedding service over the network.
- **The frontend** (Streamlit) talks only to the query API over HTTP.

---

## 2. Prerequisites

- [uv](https://docs.astral.sh/uv/) for Python dependency management (no `pip` for the backend).
- Python 3.12+.
- Accounts / API keys: **Sarvam**, **OpenRouter**, **Exa**, **Qdrant Cloud**, **Neo4j AuraDB**.
- A GPU VM (NVIDIA, ~L40S 48 GB) for ingestion and the embedding service.

---

## 3. Environment file (`.env`)

`.env` lives in `backend/` and is gitignored. Copy the template and fill it in:

```bash
cd backend
cp .env.example .env
```

### Required on every machine that runs backend code

| Key | Purpose |
|---|---|
| `SARVAM_API_KEY` | router, synthesis, graph extraction, eval judge |
| `OPENROUTER_API_KEY` | reranking (degrades to RRF order if absent) |
| `EXA_API_KEY` | live web search (validated at startup even if unused) |
| `QDRANT_URL`, `QDRANT_API_KEY` | Qdrant Cloud |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Neo4j AuraDB |
| `NEO4J_DATABASE` | AuraDB instance id — find it with `SHOW DATABASES`. Wrong value → `DatabaseNotFound` |

### Query-API machine also needs

| Key | Purpose |
|---|---|
| `EMBEDDING_SERVICE_URL` | the embedding service URL (`http://localhost:8081` on the VM) |
| `EMBEDDING_SERVICE_TOKEN` | shared secret; must match the embedding service's value |

### GPU VM also needs

| Key | Value |
|---|---|
| `EMBEDDING_SERVICE_TOKEN` | same value as above |
| `EMBEDDING_DEVICE` | `cuda:0` |
| `LLM_EXTRACT_REASONING_EFFORT` | `medium` (keep — `low`/`null` produce a sparse graph) |

Every other key is optional and uses a documented default when left blank.
`.env.example` lists and comments all of them.

---

## 4. One-time setup

### GPU VM

```bash
nvidia-smi                                       # confirm the GPU + CUDA show up
curl -LsSf https://astral.sh/uv/install.sh | sh && exec $SHELL
git clone <your-repo-url> graph && cd graph/backend
uv sync --group gpu                              # CUDA torch + FlagEmbedding + docling
export HF_HOME=$HOME/.cache/huggingface          # add to ~/.bashrc so BGE-M3 caches
# create backend/.env (section 3)
```

Confirm GPU deps import:

```bash
uv run python -c "import FlagEmbedding; print('FlagEmbedding OK')"
uv run python -c "import torch; print('cuda?', torch.cuda.is_available())"
```

> **Common error:** `ModuleNotFoundError: No module named 'FlagEmbedding'` means
> you ran `uv sync` instead of `uv sync --group gpu`. The GPU packages live in the
> `gpu` dependency group.

### CPU box (only if the query API runs separately from the VM)

```bash
cd backend
uv sync          # CPU deps only — never installs torch/FlagEmbedding/docling
# create backend/.env (section 3)
```

---

## 5. Static checks (optional, any machine)

Proves backend logic with every vendor mocked — no GPU, no cloud:

```bash
cd backend
uv run ruff check
uv run ruff format --check
uv run pyrefly check
uv run pytest          # unit suite (integration tests excluded by default)
```

---

## 6. Start the embedding service (GPU VM)

Both FastAPI apps are factories, so `--factory` is required.

```bash
cd ~/graph/backend
uv run uvicorn "src.indexing.embedding_service:create_embedding_service_app" --factory \
  --host 0.0.0.0 --port 8081
```

First start downloads BGE-M3 (multi-GB, then cached). Verify:

```bash
curl -H "Authorization: Bearer <EMBEDDING_SERVICE_TOKEN>" http://localhost:8081/health
# -> {"status":"ok","device":"cuda:0"}
```

Keep it running (tmux / systemd).

---

## 7. Ingest the corpus (GPU VM)

Ingestion reads everything in `data/raw/`. The `--sop-version` and `--doc-title`
flags stamp **every file in that folder for the run**, so ingest one file per
pass. **Order matters** — v2.1 before v3.0 so the `SUPERSEDES` edge forms.

### Clean slate (recommended before a fresh ingest)

```bash
cd ~/graph/backend
uv run python scripts/reset_stores.py --yes            # wipes Qdrant + Neo4j
# single store: --only qdrant   or   --only neo4j
```

### Four passes (do NOT reset between them)

```bash
# 1. Expense Approval v2.1 (older) — share the SAME --doc-title with v3.0
rm -f data/raw/*.md; cp data/samples/expense_approval_sop_v2.md data/raw/
uv run python scripts/run_ingestion.py \
  --doc-type SOP --sop-version v2.1 --doc-title "Expense Approval SOP" --effective-date 2024-03-01

# 2. Expense Approval v3.0 (current; creates SUPERSEDES v3.0 -> v2.1)
rm -f data/raw/*.md; cp data/samples/expense_approval_sop_v3.md data/raw/
uv run python scripts/run_ingestion.py \
  --doc-type SOP --sop-version v3.0 --doc-title "Expense Approval SOP" --effective-date 2025-06-01

# 3. Onboarding SOP
rm -f data/raw/*.md; cp data/samples/onboarding_sop_v1.md data/raw/
uv run python scripts/run_ingestion.py --doc-type SOP --sop-version v1.4 --effective-date 2024-09-15

# 4. Product info (no version flag)
rm -f data/raw/*.md; cp data/samples/product_atlas_router.md data/raw/
uv run python scripts/run_ingestion.py --doc-type product_info
```

Each pass prints a JSON summary. Healthy run: `chunks_created > 0`,
`failed_documents: []`, and for doc-bearing passes `graph_nodes_created` /
`graph_edges_created` roughly 10–14. A few `graph_extraction_empty_reply` lines on
heading/overview chunks are normal.

### (Optional) Multimodal: tables + chart captions

Needed only for the table/chart questions and the +graph+multimodal eval stage.
Two options for the captioning VLM:

**A — Hosted VLM via OpenRouter (no local GPU process).** Set in `.env`:

```bash
DOCLING_VLM_URL=https://openrouter.ai/api/v1/chat/completions
DOCLING_VLM_TOKEN=<your OpenRouter API key>
DOCLING_VLM_MODEL=<a vision-capable model slug, e.g. qwen/qwen3-vl-8b-instruct>
DOCLING_EXTRACT_CHARTS=false
```

**B — Local vLLM** (GPU-heavy; see `docs/VERIFICATION.md §6a`).

Then ingest the table/chart PDF (do NOT reset — this adds to the existing data):

```bash
mkdir -p data/samples_tables && cp data/samples/chart.pdf data/samples_tables/
INGESTION_DATA_DIR=data/samples_tables uv run python scripts/run_ingestion.py --doc-type product_info
# expect tables_extracted > 0, images_extracted > 0
```

> Keep `DOCLING_EXTRACT_CHARTS=false`: setting it `true` pulls an ~8 GB local
> Granite vision model that competes for VRAM; the remote VLM already captions
> charts as `image_caption` chunks.

### (Optional) Verify the graph

In the Neo4j / Aura browser:

```cypher
MATCH (new:Process)-[:SUPERSEDES]->(old:Process)
RETURN new.sop_version AS current, old.sop_version AS superseded, old.is_current_version AS old_flag;
-- expect current="v3.0", superseded="v2.1", old_flag=false
```

---

## 8. Start the query API

On the GPU VM (simplest) or a CPU box that can reach the embedding service:

```bash
cd ~/graph/backend
uv run uvicorn "src.api.main:create_app" --factory --host 0.0.0.0 --port 8002
```

Verify health — all three components must be `ok`:

```bash
curl http://localhost:8002/health
# status ok; qdrant ok, neo4j ok, embedding_service ok
```

### Smoke-test the API directly

```bash
# graph traversal
curl -X POST http://localhost:8002/query -H "Content-Type: application/json" \
  -d '{"question":"Which role owns the Finance Approval step of Expense Approval?"}'

# superseded-version warning (expect non-null version_warning)
curl -X POST http://localhost:8002/query -H "Content-Type: application/json" \
  -d '{"question":"In the older v2.1 expense approval SOP, what was the manager review approval threshold?"}'
```

Each response carries `answer` with `[n]` markers, `citations[]`, `tool_trace`,
and a server-issued `conversation_id`.

### Expose the API publicly (for a hosted frontend)

A hosted Streamlit frontend needs a public HTTPS URL. For a quick demo:

```bash
cloudflared tunnel --url http://localhost:8002
# -> prints an https://<random>.trycloudflare.com URL
```

> The API has no auth — anyone with the URL spends your paid credits. Fine for a
> short demo; tear the tunnel down afterwards.

### Admin endpoints — ingest a document and reset the stores

Two admin endpoints let the frontend upload a document for ingestion and wipe the
data stores. **They require the GPU host** (ingestion imports Docling + BGE-M3) and
are **intentionally unauthenticated** for the demo — a known risk, since anyone with
the URL can trigger a GPU job or delete the corpus.

**Ingest a document (async job).** Upload runs on the GPU-VM API and executes in the
background; only one ingestion runs at a time (a second returns 409).

```bash
# Start a job — returns 202 with a job_id
curl -X POST http://localhost:8002/documents \
  -F "file=@data/samples/onboarding_sop_v1.md" \
  -F "doc_type=SOP" \
  -F "sop_version=v1.4" \
  -F 'doc_title=Employee Onboarding SOP' \
  -F "effective_date=2024-09-15"
# -> {"job_id":"<uuid>","status":"pending"}

# Poll until done
curl http://localhost:8002/documents/jobs/<uuid>
# status: pending -> running -> succeeded (with "summary") | failed (with "error")
```

Form fields: `file` (required, `.pdf`/`.docx`/`.md`), `doc_type` (required,
`SOP`|`product_info`), and for SOPs the optional `sop_version`, `doc_title`,
`effective_date` (ISO `yyyy-mm-dd`). Oversized or wrong-type files return 422.

> The uploaded file is ingested on its own — unlike the manual batch in section 7,
> which ingests every file in `data/raw/`. To link a new SOP version into the
> `SUPERSEDES` chain, pass the same `doc_title` the earlier version used.

**Reset the stores (destructive).** Requires `confirm: true`; `scope` selects which
stores to wipe.

```bash
curl -X POST http://localhost:8002/admin/reset -H "Content-Type: application/json" \
  -d '{"scope":"all","confirm":true}'
# scope: "all" | "qdrant" | "neo4j"
# -> {"scope":"all","results":[{"store":"qdrant","status":"reset",...}, ...]}
```

---

## 9. Run the frontend (Streamlit)

The frontend lives in `frontend/` and is a standalone Streamlit app with its own
`requirements.txt` (separate from the uv-managed backend). See
[`frontend/README.md`](frontend/README.md) for full detail.

### Locally

```bash
cd frontend
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml     # then edit
streamlit run streamlit_app.py
```

Set `API_BASE_URL` in `.streamlit/secrets.toml` — your public tunnel URL, or
`http://127.0.0.1:8002` if testing on the same machine as the API:

```toml
API_BASE_URL = "https://<your-api-url>"
```

Streamlit opens `http://localhost:8501`. The sidebar shows a health badge and
four views: **Chat**, **Ingestion status**, **Reports**, and **Admin**.

The **Admin** view drives the two admin endpoints from section 8:

- **Ingest a document** — upload a `.pdf`/`.docx`/`.md`, choose `doc_type`, and
  (for SOPs) fill `sop_version` / `doc_title` / `effective_date`. The view uploads,
  then polls the job across reruns and shows the summary or error. A 409 means an
  ingestion is already running — wait and retry.
- **Reset data stores** — pick a scope (All / Qdrant / Neo4j) and pass the explicit
  confirmation gate before the wipe runs.

> Both admin actions need the query API running on the GPU VM (ingestion needs the
> GPU), and `API_BASE_URL` must point at that API. Avoid running a heavy ingest at
> the same moment you demo queries — the GPU-VM API serves one ingestion at a time.

### Deploy to Streamlit Community Cloud

1. Push the repo to GitHub.
2. Create an app on [share.streamlit.io](https://share.streamlit.io/) pointing at
   the repo; set the main file path to `frontend/streamlit_app.py` (Python 3.12).
3. In the app's **Secrets** dashboard add (TOML):
   ```toml
   API_BASE_URL = "https://<your-public-api-url>"
   ```
4. Deploy. The query API must be publicly reachable (HTTPS) from the Cloud runtime.

---

## 10. Run the evaluation (offline RAGAS + ablation)

Run from `backend/` on a machine that can reach the embedding service and the
cloud stores. **The stores must already be ingested** — eval scores retrieval
against whatever is in Qdrant/Neo4j.

The eval reuses existing credentials (judge = `SARVAM_API_KEY`, embeddings =
`EMBEDDING_SERVICE_URL`). All `EVAL_*` keys are optional with sensible defaults.
The one worth setting consciously:

| Key | Default | Set when |
|---|---|---|
| `EVAL_CORPUS_INCLUDES_MULTIMODAL` | `false` | `true` only if you ingested `chart.pdf` via the multimodal path |

### Smoke test first (cheap — confirms judge auth + wiring)

```bash
cd ~/graph/backend
uv run python scripts/run_eval.py --stages naive --limit 3
```

### Full run

```bash
uv run python scripts/run_eval.py
# text-only corpus (skip the multimodal stage):
uv run python scripts/run_eval.py --stages naive,hybrid_rerank,graph
```

Flags: `--stages`, `--golden-set <path>`, `--output-dir <path>`, `--limit <n>`.

### Results

Two files land in `EVAL_OUTPUT_DIR` (default `data/eval/reports/`):

```
eval_report_<timestamp>.json   # structured ablation report
eval_report_<timestamp>.md     # markdown table (one row per stage + deltas)
```

Commit the markdown file to have the frontend **Reports** view render real numbers.
Look for faithfulness and context precision rising from naive → +hybrid+rerank →
+graph. A non-monotonic curve with ~34 items is a finding to discuss, not a bug.

---

## 11. What runs where (quick reference)

| Component | Machine | Command | Port |
|---|---|---|---|
| Embedding service | GPU VM | `uvicorn src.indexing.embedding_service:create_embedding_service_app --factory` | 8081 |
| Ingestion (batch) | GPU VM | `python scripts/run_ingestion.py ...` | — |
| VLM (optional, local) | GPU VM | `vllm serve <model>` | 8000 |
| Query API | GPU VM / CPU box | `uvicorn src.api.main:create_app --factory` | 8002 |
| Admin ingest/reset | GPU VM (via query API) | `POST /documents`, `POST /admin/reset` | 8002 |
| Frontend | local / Streamlit Cloud | `streamlit run streamlit_app.py` | 8501 |
| Evaluation (offline) | GPU VM / CPU box | `python scripts/run_eval.py` | — |

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `uvicorn ... Attribute "app" not found` | Use `--factory` and the factory target (`src.api.main:create_app`) |
| `ModuleNotFoundError: No module named 'FlagEmbedding'` | Run `uv sync --group gpu` on the VM |
| `/health` embedding_service `unhealthy: EMBEDDING_SERVICE_URL not configured` | Query API started outside `backend/`; restart from `backend/` |
| `/health` 503 | Qdrant or embedding service down; resume a paused AuraDB/Qdrant instance |
| `Neo.ClientError.Database.DatabaseNotFound` | `NEO4J_DATABASE` wrong; run `SHOW DATABASES`, use the instance id |
| No `SUPERSEDES` edge / no version warning | v2.1 and v3.0 ingested under different titles or wrong order — same `--doc-title`, v2.1 first |
| Reranking logs `rerank_fallback_to_rrf` | `OPENROUTER_API_KEY` missing/placeholder; answers still work without rerank |
| `tables_extracted`/`images_extracted` stay 0 | Source is Markdown (no tables/pictures) — use a PDF like `chart.pdf` |
| Eval: "vector store is not ready" | Qdrant empty/unreachable; ingest first |
| Eval: widespread `n/a` scores | Judge calls failing; check the report's Failures section (auth, model, or raise `EVAL_JUDGE_MAX_TOKENS`) |
| Frontend: "API_BASE_URL is not set" | Add it to `.streamlit/secrets.toml` locally or the Secrets dashboard on Cloud |
| Admin ingest: 409 "already running" | One ingestion runs at a time; wait for the active job to finish |
| Admin ingest: job `failed` with a GPU-deps error | The query API is on a CPU box without the `gpu` group; ingestion needs the GPU VM |
| Admin reset: 422 "requires confirm=true" | Send `{"confirm": true}` in the body (the frontend's confirmation gate does this) |
