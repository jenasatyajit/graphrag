# Graph-Augmented Multimodal Company-Knowledge QA

A retrieval-augmented question-answering backend over internal company documents
(SOPs and product sheets). It combines hybrid vector search, reranking, a
knowledge graph, multimodal ingestion (tables + image captions), a tool-calling
router with live web search, conversation memory, and citation-grounded answer
synthesis. Every factual claim in an answer carries a citation; uncited claims
are dropped. The engineering plan lives in [`docs/BACKEND.md`](docs/BACKEND.md)
and the architecture in [`docs/SPEC.md`](docs/SPEC.md).

## Architecture at a glance

- `POST /query` runs a Sarvam tool-calling router that chooses between internal
  retrieval, live web search, and a direct reply, enforces routing rules as
  code-level guardrails, then synthesizes one grounded, cited answer.
- Retrieval is hybrid (dense + sparse RRF in Qdrant) + OpenRouter rerank + graph
  facts from Neo4j. Web results are labeled `external` and are never persisted.
- Graceful degradation is a feature: a reranker, graph, or web outage lowers
  answer quality but still returns a valid, correctly-labeled response. Only a
  true dead end (internal retrieval down with no web fallback) returns a 503.

## Two-machine topology

This project runs across two machines plus cloud services (BACKEND.md §2.2).
The split is deliberate: BGE-M3 embeddings and Docling's vision model want a
GPU; the API backend does not.

- **CPU API box** — runs the FastAPI service (`/query`, `/ingest`, `/health`):
  router, retrieval, rerank, synthesis. It calls out to Sarvam, OpenRouter, Exa,
  Qdrant Cloud, Neo4j AuraDB, and the GPU VM's embedding service for query-time
  embedding. It must never import `torch`/`FlagEmbedding` (GPU-only deps live in
  the `gpu` dependency group).
- **GPU VM (NVIDIA L40S)** — hosts the BGE-M3 embedding service and runs
  `scripts/run_ingestion.py` in-process (Docling parse + VLM caption → BGE-M3
  embed → writes chunks to Qdrant Cloud and the graph to Neo4j AuraDB).

Cloud services (Qdrant, Neo4j, Sarvam, OpenRouter, Exa) are shared by both.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) for dependency management (no `pip`).
- Python 3.12+.
- Accounts/keys for Sarvam, OpenRouter, Exa, Qdrant Cloud, and Neo4j AuraDB.
- A GPU VM only for ingestion and the embedding service (not for the API).

## Quickstart

All commands below run from the `backend/` directory.

### 1. Install dependencies

```bash
cd backend
uv sync            # API/runtime + dev deps (CPU-only, no GPU packages)
```

On the **GPU VM only**, also install the GPU group for ingestion and the
embedding service:

```bash
uv sync --group gpu   # adds docling, FlagEmbedding, torch (VM only)
```

### 2. Configure the environment

```bash
cp .env.example .env
```

Fill in the required keys in `.env`: `SARVAM_API_KEY`, `OPENROUTER_API_KEY`,
`EXA_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `NEO4J_URI`, `NEO4J_USER`,
`NEO4J_PASSWORD`, and — for the query API — `EMBEDDING_SERVICE_URL` (the GPU VM
embedding service). Every other key is optional; a blank value falls back to the
documented default. `.env.example` lists and comments every available setting.

### 3. Ingest documents (GPU VM)

Ingestion is an offline batch job that runs on the GPU VM, reading from the
directory configured by `INGESTION_DATA_DIR` (default `data/raw/`):

```bash
# SOPs (pass the version so superseded-version edges are created correctly)
uv run python scripts/run_ingestion.py --doc-type SOP --sop-version v3.0

# Product info sheets
uv run python scripts/run_ingestion.py --doc-type product_info
```

Optional flags: `--effective-date YYYY-MM-DD`. The run writes a machine-readable
summary to `INGESTION_SUMMARY_PATH`, which `POST /ingest` reports back.

### 4. Run the API server (CPU box)

```bash
uv run uvicorn src.api.main:create_app --factory --reload
```

`src.api.main` exposes an application factory (`create_app`), so uvicorn is
pointed at it with `--factory` rather than at a module-level `app`. This keeps
settings validation and client construction inside app startup instead of at
import time.

- `GET /health` — liveness for Qdrant, Neo4j, and the embedding service.
- `POST /query` — `{ "question": "...", "conversation_id": null }`.
- `POST /ingest` — returns the latest ingestion summary (does not start a job).

### 5. Run the evaluation (offline)

The RAGAS + four-stage ablation harness (naive → hybrid → +graph → +multimodal)
reads the curated golden set and writes JSON + markdown reports:

```bash
uv run python scripts/run_eval.py
```

Useful overrides: `--stages naive,hybrid`, `--golden-set <path>`,
`--output-dir <path>`, `--limit <n>`. The judge is a Sarvam model via its
OpenAI-compatible endpoint; AnswerRelevancy embeddings reuse the query-time
embedding service, so the eval path stays CPU-only.

## Testing

From `backend/`:

```bash
uv run pytest              # unit suite (all vendors mocked; integration excluded)
uv run ruff check          # lint
uv run ruff format --check # formatting
uv run pyrefly check       # type checking
```

The default `pytest` run excludes live-service tests via
`addopts = -m 'not integration'`. To run the opt-in integration suite (which
calls live external services and may consume API credits), run it explicitly:

```bash
uv run pytest -m integration
```

Do not run integration tests in a tight loop — Exa, OpenRouter, and Sarvam are
metered (SPEC §13).
