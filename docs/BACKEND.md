# BACKEND.md — Build Guide for AI Coding Agents

> **Audience:** AI coding agents implementing the backend of this project from scratch.
> **Read `docs/SPEC.md` first, then this file.** SPEC.md is the *product/architecture* source of truth (what to build and why). This file is the *engineering execution plan* (how to build it, in what order, and when a phase is "done").
>
> **Golden rule when the two disagree:** SPEC.md wins on architecture, schemas, and behavior. This file wins on process, tooling, testing, and quality bars. If either conflicts with a live SDK/API, the live docs win — but you must record the deviation (see [§3 Verified API Facts](#3-verified-api-facts-checked-against-live-docs)).

---

## 1. How to Use This Document

1. Work **phase by phase**, top to bottom. Do not start a phase until the previous phase's **Definition of Done (DoD)** is fully met.
2. Every phase has: **Goal → Tasks → Tests to write → Definition of Done**. All four are mandatory.
3. Before writing integration code for any external service (Sarvam, Cohere, Exa, Qdrant, Neo4j, Docling, RAGAS, BGE-M3), **re-verify the current API** using web search. Treat every code snippet in SPEC.md and in this file as *intent*, not a verified signature.
4. **Never assume.** If a requirement is ambiguous, missing, or contradicts reality, stop and record a question in [§12 Open Questions](#12-open-questions-ask-do-not-assume). Do not silently guess.
5. Commit at the end of every phase with a passing test suite, passing lint, and passing type checks.
6. Keep this file honest: when you discover an API has changed, update [§3](#3-verified-api-facts-checked-against-live-docs) with the corrected fact and the date you verified it.

---

## 2. Context (What You Are Building)

A Question-Answering backend over internal company documents (SOPs + product info) that fuses:

- **Hybrid vector search** (dense + sparse via BGE-M3, fused with RRF in Qdrant),
- a **domain knowledge graph** (Neo4j) for relational / multi-hop / SOP-version questions,
- **multimodal ingestion** (tables → markdown, images/charts → captions),
- **routed live web search** (Exa) for genuinely-external questions,
- with **every claim citation-grounded** and tagged by trust tier (`internal` vs `external`), and a **version-conflict warning** when a superseded SOP would be cited.

The deliverable also includes a **RAGAS + ablation** evaluation harness (naive → hybrid → +graph → +multimodal) for an academic write-up.

The backend is a Python service (`backend/`), packaged with **uv**, exposing a small **FastAPI** surface (`POST /query`, `POST /ingest`, `GET /health`). Ingestion is an **offline batch script**, separate from the query path.

The full repository layout, data schemas (chunk, graph, citation), API contract, prompt templates, and success criteria are defined in **SPEC.md §3–§11**. Do not restate them here; implement against them.

### 2.1 Non-negotiable behaviors (from SPEC, restated because they drive tests)

- Generation must **refuse to emit an uncited claim** (SPEC §6.3). This is a hard rule and a required test.
- The router must **cap tool calls** at `MAX_AGENT_ITERATIONS` and return a "couldn't find a confident answer" response rather than hallucinate (SPEC §7.4).
- Web results are **never persisted** into the internal chunk store; they live only in the per-request object (SPEC §7.3).
- The reranker must **degrade gracefully to RRF order** on Cohere failure (SPEC §7.3, §12, §13).
- Superseded SOP versions must surface a **version warning** (SPEC §7.5).

### 2.2 Deployment Topology (decided — implement to this shape)

This project runs across **two machines** plus **cloud services**. This split is deliberate: BGE-M3 and the Docling VLM want a GPU, but the API backend does not.

```
┌─────────────────────────────┐         ┌──────────────────────────────────────────┐
│  API backend (your box /     │  HTTP   │  GPU VM  (Ubuntu 22.04, NVIDIA L40S 48GB)  │
│  small server)               │────────▶│                                            │
│  - FastAPI: /query /ingest*  │ embed   │  - embedding_service (FastAPI + BGE-M3)    │
│    /health                   │ request │      GET /health  POST /embed              │
│  - router, retrieval,        │         │  - run_ingestion.py runs HERE in-process:  │
│    rerank, synthesis         │         │      Docling parse (+VLM caption) →        │
│  - calls Sarvam/Cohere/Exa   │         │      BGE-M3 embed → writes to cloud stores │
└───────────┬─────────────────┘         └───────────────┬────────────────────────────┘
            │                                            │
            │ Qdrant + Neo4j (cloud)  ◀──────────────────┘  (ingestion writes here)
            ▼
   ┌───────────────────────────────┐
   │ Qdrant Cloud   Neo4j AuraDB    │   Sarvam API   Cohere API   Exa API  (all cloud)
   └───────────────────────────────┘
```

**Decisions locked in:**
- **Q-A → (a):** BGE-M3 is exposed as a **small FastAPI HTTP service on the GPU VM** (`embedding_service`), wrapping `BGEM3FlagModel`. It returns **both dense and sparse** vectors in one response. The API backend calls it for **query-time** embedding (embedding the incoming question).
- **Q-B → ingestion-on-VM:** `scripts/run_ingestion.py` runs **on the GPU VM**, importing Docling + BGE-M3 **in-process** (no HTTP hop for the batch job), writing chunks to **cloud Qdrant** and graph to **cloud Neo4j**. Only the batch job runs on the VM; the query API does not.
- **Q-C:** VM is **Ubuntu 22.04 + NVIDIA L40S (48 GB)**. Plenty of VRAM for BGE-M3 (~2 GB) and a hosted-style Docling VLM caption model simultaneously.
- **Q-1 → hosted Docling captioning:** image/chart captions come from Docling's picture-description pipeline pointed at an OpenAI-compatible VLM endpoint (see [§3.5](#35-docling-parsing--multimodal), [§3.10](#310-gpu-vm-embedding-service--ingestion-host)).
- **Q-2:** RAGAS judge = a **Sarvam model** via the OpenAI-compatible/LiteLLM path (record exact model in the eval report).
- **Q-3:** type checker = **pyrefly**.
- **Q-4:** **Qdrant Cloud** + **Neo4j AuraDB** (both cloud). `.env` URIs point at cloud; `/health` verifies both.

**Implications the code must honor:**
- `embeddings.py` must support **two modes** behind one interface: **in-process** `BGEM3FlagModel` (used by ingestion on the VM) and **HTTP client** to the embedding service (used by the query API). Same dense+sparse output contract either way, so hybrid search is identical.
- The embedding **request/response contract** is a shared schema (dense: `list[float]`; sparse: `{indices: list[int], values: list[float]}`, indices ascending). Define it once (Pydantic) and reuse on both sides.
- The API backend must **never** import `FlagEmbedding`/`torch` — keep GPU-only deps out of its dependency closure (put them in a `gpu`/`ingestion` dependency group, not the default). This keeps the API image small and CPU-only.
- Network calls to the embedding service get the same treatment as any external call: timeout, retry-with-backoff, and a clear error if the VM is unreachable (query-time embedding failure → 503 with a clean message).

---

## 3. Verified API Facts (checked against live docs)

> These were verified against live documentation on **2026-09-13**. SDKs move fast — **re-verify before wiring each one**, and update this table (with a new date) if anything changed. Where a fact here contradicts SPEC.md, this section is the corrected reality; keep SPEC.md's *intent* but implement the corrected fact and note it in a code comment.

### 3.1 Sarvam LLM (router + generation)
- **OpenAI-compatible** chat completions at `https://api.sarvam.ai/v1/chat/completions`. Auth header is `api-subscription-key` (not a bearer token) when calling directly; the `sarvamai` SDK wraps this.
- **Model:** `sarvam-105b` is the current chat model. ⚠️ **`sarvam-30b` and `sarvam-m` are deprecated** and rejected by the API. SPEC still lists `sarvam-30b` in some tables — use **`sarvam-105b` only**.
- Supports **tool calling** (`tools`, `tool_choice`; `type: "function"` only) and **`reasoning_effort`**.
- **`reasoning_effort`** accepts `low | medium | high`, and reasoning can be **disabled by passing `None`/null**. Reasoning is **on by default** and reasoning tokens consume `max_tokens` — so set `reasoning_effort=None` for the router and set a generous `max_tokens` on synthesis. (Matches SPEC §13.)
- 128K context window. `max_tokens` ceiling depends on plan tier — do not hardcode a large value blindly; make it configurable.

### 3.2 Cohere Rerank
- Endpoint: `POST https://api.cohere.com/v2/rerank` (v2 API). Python SDK: `cohere.ClientV2(...).rerank(...)`.
- ⚠️ **Current model is `rerank-v4.0-pro`** (or `rerank-v4.0-fast` for lower latency). SPEC's `rerank-v3.5` still works but is a prior generation — **default to `rerank-v4.0-pro`** and keep it configurable via `RERANK_MODEL`.
- Request: `model`, `query`, `documents` (list of strings; YAML strings for structured data), `top_n`, optional `max_tokens_per_doc` (default 4096). Recommended max ~1000 docs/request.
- Reuse a single client. Wrap with try/except + timeout; on failure fall back to RRF order.

### 3.3 Exa web search
- SDK: `exa-py`; `from exa_py import Exa`. Primary method: `exa.search(query, type="auto"|"fast", num_results=..., contents={...})`.
- On `/search`, content options (`text`, `highlights`, `summary`) are **nested under `contents`**; use `contents={"highlights": True}` as the retrieval default. For known URLs use `exa.get_contents([...])` where those are top-level args.
- ⚠️ The standalone **Research API is deprecated**; the old `context` content option is deprecated (use `highlights`/`text`).
- Attach `retrieved_at` (UTC ISO 8601) and the result `url` to every result at your layer. Respect the monthly credit cap — do not call Exa in tight test loops (mock it in tests).

### 3.4 Sarvam Vision — IMPORTANT ARCHITECTURE CORRECTION
- ⚠️ **Sarvam Vision is a document OCR / Document-AI service, NOT a general image-captioning VLM.** It is exposed via the async **Document AI** job API (`doc_ai.digitise` / `doc_ai.extract`: create job → poll → fetch), max 10 pages/PDF, 200 MB/file, ~10 req/min. It returns OCR'd text/tables (HTML/Markdown/JSON), *not* free-form "describe this chart" captions.
- SPEC §7.1 assumes Sarvam Vision can caption arbitrary images/charts. **This is not how the API works.** **Decision (2026-09-13): use hosted Docling picture description ([§3.5](#35-docling-parsing--multimodal)) for image/chart captioning** — a hosted VLM via Docling's OpenAI-compatible `PictureDescriptionApiOptions`. Sarvam Vision's Doc-AI OCR may still be used for scanned/OCR-only PDFs if needed, but it is **not** the captioner. See [D-3](#11-deviations-log-keep-this-current).

### 3.5 Docling (parsing + multimodal)
- Entry point: `DocumentConverter().convert(source).document`. Configure via `PdfPipelineOptions` + `PdfFormatOption` keyed by `InputFormat.PDF`.
- Tables and pictures are first-class: `doc.tables`, `doc.pictures`; export tables to markdown; page/provenance metadata is available on items.
- **Built-in picture description** (captioning): set `pipeline_options.do_picture_description = True`. For a remote OpenAI-compatible VLM use `PictureDescriptionApiOptions(url=..., params={"model": ...}, prompt=..., timeout=...)` **and** `pipeline_options.enable_remote_services = True`. A local default (SmolVLM) also exists.
- **Chart extraction**: `pipeline_options.do_chart_extraction = True` converts bar/pie/line charts to tabular data.
- **This replaces the bespoke `multimodal.py` VLM-call design** in SPEC where possible — prefer Docling's pipeline for captioning; keep `multimodal.py` as a thin wrapper/orchestrator. Record this deviation in code comments.

### 3.6 BGE-M3 embeddings (dense + sparse)
- `from FlagEmbedding import BGEM3FlagModel`; `model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)`.
- Output keys: `dense_vecs` (1024-dim) and `lexical_weights` (token_id → weight dict). Convert lexical weights to Qdrant sparse `indices`/`values` (indices ascending). Do **not** drop the sparse output (SPEC §7.2).
- `use_fp16=True` speeds up with slight quality loss. First load downloads a multi-GB model — cache it; make the device configurable (CPU fallback).

### 3.7 Qdrant hybrid search
- Create a collection with a **named dense vector** and a **named sparse vector** (`sparse_vectors_config={"<name>": models.SparseVectorParams(modifier=models.Modifier.IDF)}`).
- Query with `client.query_points(collection, prefetch=[Prefetch(dense...), Prefetch(sparse...)], query=models.FusionQuery(fusion=models.Fusion.RRF), limit=..., with_payload=True)`. `models.RrfQuery(rrf=models.Rrf(k=..., weights=[...]))` allows tuning. Requires Qdrant ≥ 1.17 and a matching client.
- Return ~20 candidates before rerank (SPEC §7.3).

### 3.8 Neo4j Python driver
- Current major version is **6.x**; install the `neo4j` package (`neo4j-driver` is deprecated). `GraphDatabase.driver(uri, auth=(user, pw))`; call `driver.verify_connectivity()` at startup (health check).
- Prefer `driver.execute_query(cypher, param=..., database_="neo4j", routing_="r"|"w")` for simple cases. **Always pass parameters** (never string-concatenate — Cypher injection). Always specify `database_`.

### 3.9 RAGAS
- Current line is **v0.4.x** with a new **collections-based** metrics API: `from ragas.metrics.collections import Faithfulness, ContextPrecision, ContextRecall, AnswerRelevancy`; metrics return a `MetricResult` (`.value`, optional `.reason`) via `await metric.ascore(**kwargs)`.
- The legacy API (`from ragas.metrics import Faithfulness`, `SingleTurnSample`, `evaluate()`) still works with deprecation warnings. **Pick one API and pin the RAGAS version** so behavior is stable for the ablation write-up.
- Judge LLM: use `ragas.llms.llm_factory` (v0.4) routed via LiteLLM to Sarvam's **OpenAI-compatible** endpoint (decision Q-2). Record the exact Sarvam model used as judge in the eval report — it affects reported scores.

### 3.10 GPU VM: embedding service + ingestion host

> Target: **Ubuntu 22.04, NVIDIA L40S 48 GB.** This subsection is setup guidance for the VM; re-verify driver/CUDA versions against current NVIDIA/PyTorch docs at install time.

**One-time VM setup:**
1. **NVIDIA driver + CUDA:** install a recent NVIDIA driver for the L40S and verify with `nvidia-smi` (should list the L40S and a CUDA version). PyTorch ships its own CUDA runtime via wheels, so a system CUDA *toolkit* is usually not required — the driver is what matters.
2. **uv:** `curl -LsSf https://astral.sh/uv/install.sh | sh` then restart the shell.
3. **Project:** clone the repo; from `backend/` run `uv sync --group gpu` (the `gpu` group carries `FlagEmbedding`, `torch`, `docling`, and the embedding-service deps; see below). uv provisions the venv and downloads a CUDA-enabled `torch` wheel.
4. **Model cache:** first BGE-M3 load downloads a multi-GB model from Hugging Face. Set `HF_HOME`/`HUGGINGFACE_HUB_CACHE` to a persistent path so it isn't re-downloaded. Pre-warm by loading the model once.

**BGE-M3 in-process (used by `run_ingestion.py` on the VM):**
```python
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices=["cuda:0"])
out = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
# out["dense_vecs"]  -> ndarray [n, 1024]
# out["lexical_weights"] -> list of {token_id: weight}; convert to Qdrant sparse (indices ascending)
```
`use_fp16=True` is fine on the L40S. On CUDA it is fast; keep a CPU fallback path (`devices=["cpu"]`) for local dev/tests but never rely on it for the real corpus.

**Embedding service (used by the query API):** a minimal FastAPI app on the VM that loads the model **once at startup** (`lifespan`) and serves:
- `GET /health` → 200 when the model is loaded (and reports device).
- `POST /embed` → body `{"texts": [...], "mode": "query"|"passage"}`; returns the shared contract: `{"dense": [[...]], "sparse": [{"indices": [...], "values": [...]}]}`. Batch inside the handler; cap batch size; add a small timeout guard.
- Protect it: bind to a private network/VPN or put an auth token/reverse proxy in front — it should **not** be open to the public internet.
- Run it with `uv run uvicorn <module>:app --host 0.0.0.0 --port 8081`. For durability, run under **systemd** (or tmux for a quick demo) — the agent should NOT start it via a blocking foreground command; document the systemd unit instead.

**Docling on the VM:** ingestion imports Docling in-process. For **hosted captioning** (Q-1) point Docling at an OpenAI-compatible VLM:
```python
pipeline_options.enable_remote_services = True
pipeline_options.do_picture_description = True
pipeline_options.do_chart_extraction = True   # bar/pie/line charts -> tabular
pipeline_options.picture_description_options = PictureDescriptionApiOptions(
    url="<VLM /v1/chat/completions endpoint>",
    headers={"Authorization": "Bearer <token>"},   # or provider header
    params={"model": "<vlm-model-id>"},
    prompt="Describe this chart/image factually for retrieval. Be concise and specific.",
    timeout=90,
)
```
Point `url` at the self-hosted vLLM endpoint below.

**Self-hosted VLM endpoint (decided — vLLM + Qwen2.5-VL-7B-Instruct):** run a separate vLLM process on the L40S serving an OpenAI-compatible `/v1/chat/completions`:
```bash
# re-verify the exact flags against current vLLM docs at install time
uv pip install "vllm>0.7.2"   # or run the official vLLM docker image
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 \
  --limit-mm-per-prompt '{"image":2,"video":0}' \
  --gpu-memory-utilization 0.45      # leave room for BGE-M3 + ingestion on the same 48GB card
```
- **Model choice:** `Qwen/Qwen2.5-VL-7B-Instruct` — single-GPU, strong on charts/tables/figures, OpenAI-compatible via vLLM. (A 72B variant needs multi-GPU; not needed here.)
- **VRAM budget:** the 7B VLM + KV cache fits well under half the L40S; cap `--gpu-memory-utilization` so BGE-M3 (~2 GB) and Docling ingestion coexist. If you'd rather not co-locate, run the VLM on a second port/process and tune the split, or move it to a separate box.
- Docling calls this endpoint per image during ingestion; images are passed as `image_url` (base64 supported). Set `DOCLING_VLM_URL=http://localhost:8000/v1/chat/completions`, `DOCLING_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct`, and a token if you front it with auth. vLLM accepts `api_key="EMPTY"` by default — protect it at the network layer, don't expose it publicly.
- Run vLLM under **systemd** (like the embedding service); do not launch it as a blocking foreground command.

**How the API backend consumes it:** `embeddings.py` exposes one interface with two backends — `InProcessBGEM3` (VM/ingestion) and `HttpEmbeddingClient` (query API → `POST /embed`). The query API depends only on the HTTP client; it must not pull in `torch`/`FlagEmbedding`. Add `EMBEDDING_SERVICE_URL` (and an auth token) to `.env`/SPEC §5 config.

---

## 4. Global Constraints (apply to every phase)

### 4.1 Tooling & packaging
- Python **3.11+** (pin via `.python-version`). Managed with **uv** only. Never `pip install`, never `requirements.txt`.
- Add deps with `uv add <pkg>`; add dev tools with `uv add --dev <pkg>` (lands in PEP 735 `[dependency-groups]`). Run everything via `uv run ...`. Commit `uv.lock`.
- All uv commands run from inside `backend/`.

### 4.2 Code quality (enforced, not aspirational)
- **Ruff** for lint + format. Config in `pyproject.toml`. Suggested `lint.select = ["E","F","I","N","UP","B","C4","SIM","RUF"]`, `ignore = ["E501"]` (formatter owns line length). Run order: `ruff check --fix` then `ruff format`.
- **Type checking** with **Pyrefly** (decided). `uv add --dev pyrefly`; run `uv run pyrefly check`; initialize its config with `uv run pyrefly init` (writes a `[tool.pyrefly]` table). Full type hints on every function signature (SPEC §12). Treat type errors as build failures.
- **pre-commit** wiring Ruff + type-check + `uv-lock` + basic hygiene hooks (trailing whitespace, EOF, check-toml/yaml). Install with `uvx pre-commit install`.
- Zero lint errors, zero type errors, and a green test suite are part of **every** phase's DoD.

### 4.3 Configuration & secrets
- All config via `.env` / environment, loaded through a single typed **Pydantic `Settings`** object (`pydantic-settings`). No hardcoded keys, URLs, model names, or file paths anywhere (SPEC §12).
- `.env` is gitignored; `.env.example` lists every key from SPEC §5 with empty values + one-line comments. Never read or echo secret *values* in logs.
- Model names (`LLM_MODEL`, `RERANK_MODEL`, `EMBEDDING_MODEL`), iteration caps, and top-N are config, not literals.

### 4.4 Error handling & resilience (SPEC §12, §13)
- **Every** external call (Sarvam, Cohere, Exa, Qdrant, Neo4j, Docling remote, embeddings) is wrapped in try/except with a **sensible timeout** and, where transient, bounded **retry with backoff** (e.g. `tenacity`). These are network calls; they will fail.
- Define a domain exception hierarchy (e.g. `AppError` → `RetrievalError`, `GraphError`, `RerankError`, `LLMError`, `IngestionError`, `NotFoundError`, `ValidationError`). Never let a raw third-party exception reach the API boundary.
- **Graceful degradation is a feature:** reranker failure → RRF order; graph failure → vector-only with a logged warning; web-search failure → answer from internal only and note it. A dependency being down should degrade quality, not crash the request.

### 4.5 Logging & observability (SPEC §12)
- **Structured logging** with `structlog` (JSON in prod, console in dev), configured once at startup and routed through stdlib logging so uvicorn logs match.
- A **request/correlation ID** is generated (or read from `X-Request-ID`), bound to `structlog.contextvars` in middleware, echoed back on the response header, and present on every log line for that request.
- Log **every tool call, its inputs/summary of results, retrieval candidate counts, rerank scores, and token usage** (SPEC §7.4, §12). Never log secrets or full document bodies (log ids + counts).

### 4.6 User-facing errors
- Uniform JSON error envelope on all failures: `{ "error": { "code", "message", "details", "request_id" } }`. Messages are human-readable and safe (no stack traces, no secrets). Internal detail goes to logs keyed by `request_id`.
- Validation errors (bad request body) → 422 with field-level detail. Not-found → 404. Upstream/dependency failure that can't degrade → 503 with a clear "service temporarily unavailable" message. Unexpected → 500 generic message + logged traceback.

### 4.7 Testing philosophy
- Write a unit test **alongside** each new module in `src/`, not at the end (SPEC §12). Use `pytest`.
- **Mock all external services in unit tests.** Never hit Sarvam/Cohere/Exa/Qdrant/Neo4j live in the default `uv run pytest` run — it's slow, flaky, costs money, and burns Exa/Cohere credits (SPEC §13).
- Mark any test that needs live services with `@pytest.mark.integration` and keep it out of the default run (e.g. `addopts = ["-m", "not integration"]`). Integration tests are opt-in.
- Deterministic tests: seed/fixture embeddings and LLM responses. Test the *logic* (fusion, citation grounding, routing rules, version-warning) with fakes, not the vendors.
- Clean up any temp files a test creates.

### 4.8 Verify-before-done
- After each change, run `uv run ruff check`, `uv run pyrefly check`, and `uv run pytest` and make them pass before claiming the phase complete. A command exiting 0 is necessary but not sufficient — confirm the phase's concrete DoD criteria.

### 4.9 Conversation memory (multi-turn)

`POST /query` accepts `conversation_id` (SPEC §8) and this build supports **real multi-turn conversations** ([D-10](#11-deviations-log-keep-this-current)). Requirements:

- **Storage:** persist per-conversation turn history (user question + final answer, plus enough metadata to rebuild context — e.g. tool trace, whether web was used). Start with a simple, swappable store (in-memory for dev/tests behind an interface; a lightweight persistent backend — SQLite or the existing Neo4j/Qdrant infra — for real use). Do **not** invent a new heavyweight datastore without asking. Cap history length/age to bound cost.
- **Follow-up contextualization:** before retrieval, if the question is a follow-up (pronouns, ellipsis, "what about X"), **rewrite it into a standalone query** using recent history via a cheap Sarvam call (`reasoning_effort=None`). Retrieval and routing then operate on the rewritten, self-contained question. This keeps hybrid/graph retrieval accurate for terse follow-ups.
- **Prompting:** include a bounded window of prior turns in the synthesis context, clearly separated from retrieved evidence. History provides *conversational* context only — it is **not** a citation source. The uncited-claim and citation-grounding rules (SPEC §6.3) still apply to every factual claim; do not let prior-turn text substitute for a citation.
- **Isolation & privacy:** never leak one conversation's history into another; scope strictly by `conversation_id`. A missing/unknown `conversation_id` starts a new conversation. Don't log full history bodies (log ids + turn counts).
- **Statelessness of retrieval stores:** conversation memory lives in the conversation store only — it must **never** be written into the internal chunk store or the graph (same rule as web results, SPEC §7.3).

---

## 5. Phase Plan

The build order mirrors SPEC §10. Each phase must be **demoable and tested** before the next. Do not skip ahead — Phase 1 is the safety net; later phases layer onto a working end-to-end system.

> Legend: **G** = Goal · **T** = Tasks · **✅ Tests** = tests to write this phase · **DoD** = Definition of Done.

---

### Phase 0 — Scaffold & Health

**G:** A runnable, typed, linted FastAPI skeleton with config, logging, error handling, and a real health check. No RAG yet.

**T:**
1. `uv init` inside `backend/`, `src/` layout, pin Python via `.python-version`. Create the exact folder tree from SPEC §3 (empty modules with docstrings + `TODO(phase-N)` markers are fine).
2. `uv add fastapi uvicorn pydantic pydantic-settings structlog`. `uv add --dev pytest pytest-asyncio ruff pyrefly pre-commit`.
3. Configure Ruff, Pyrefly (`uv run pyrefly init`), pytest, and pre-commit in `pyproject.toml` per [§4.2](#42-code-quality-enforced-not-aspirational). Install hooks.
4. Implement typed `Settings` (pydantic-settings) loading SPEC §5 env vars **plus the topology keys** `EMBEDDING_SERVICE_URL` and `EMBEDDING_SERVICE_TOKEN` (query-time embedding, [§2.2](#22-deployment-topology-decided--implement-to-this-shape)/[§3.10](#310-gpu-vm-embedding-service--ingestion-host)), the Docling VLM caption keys `DOCLING_VLM_URL` / `DOCLING_VLM_TOKEN` / `DOCLING_VLM_MODEL` (Phase 4), and conversation-memory keys `CONVERSATION_STORE` (backend selector) + `CONVERSATION_HISTORY_MAX_TURNS` (Phase 5, [§4.9](#49-conversation-memory-multi-turn)). Write `.env.example` (every SPEC §5 key + these, empty, commented). Gitignore `.env`, `data/raw/`, `data/processed/`, model caches, `.venv`.
5. Implement `structlog` setup + request-ID middleware + the uniform error envelope + domain exception hierarchy ([§4.4](#44-error-handling--resilience-spec-12-13)–[§4.6](#46-user-facing-errors)).
6. FastAPI app via a `create_app()` factory using **`lifespan`** (not `on_event`). Wire CORS, middleware, exception handlers, and the router.
7. `GET /health`: return 200 only if it can reach Qdrant **and** Neo4j (SPEC §8). Until those clients exist, stub with clearly-marked `TODO(phase-1/phase-3)` that currently reports component status; make the check real as soon as each client lands.
8. Add thin client-wrapper stubs (`vector_store.py`, `graph_store.py`) with connection + `verify_connectivity` methods so `/health` can become real early.

**✅ Tests:**
- `create_app()` builds; `GET /health` returns expected shape.
- Settings loads from env and fails clearly on missing required keys.
- A route that raises a domain exception returns the correct status + error envelope + `request_id`.
- Request-ID middleware sets/echoes `X-Request-ID`.

**DoD:**
- `uv run uvicorn` serves the app; `/health` responds. `uv run pytest`, `uv run ruff check`, and the type checker all pass. `.env.example` complete. pre-commit installed and green. Committed.

---

### Phase 1 — Naive text-only RAG (the safety net)

**G:** True end-to-end: parse a few SOPs → chunk → embed → Qdrant dense search → Sarvam-105B answers with basic `[n]` citations. Prove `POST /query` works before adding complexity.

> **Topology note:** per [§2.2](#22-deployment-topology-decided--implement-to-this-shape), Docling + BGE-M3 live in the **`ingestion`/`gpu` dependency group** and run on the GPU VM; the API backend must not import them. Split deps accordingly from the start.

**T:**
1. Dependencies, grouped: API/runtime → `uv add qdrant-client sarvamai tenacity httpx` (the API talks to the embedding service over HTTP via `httpx`, and to Qdrant/Sarvam). GPU/ingestion group → `uv add --group gpu docling FlagEmbedding torch` (installed only on the VM). The embedding-service app (also on the VM) reuses `fastapi`/`uvicorn` + the `gpu` group. Re-verify each API ([§3](#3-verified-api-facts-checked-against-live-docs)).
2. `parser.py`: Docling → narrative text blocks with page numbers (tables/images deferred to Phase 4; extract text only now). Runs on the VM.
3. `chunker.py`: SOP step-aware chunking with fixed-size fallback for prose (SPEC §7.1). Emit chunks matching the SPEC §6.1 chunk schema (as a Pydantic model), `content_type="text"`, `source_tier="internal"`.
4. `embeddings.py`: **one interface, two backends** ([§3.10](#310-gpu-vm-embedding-service--ingestion-host)) — `InProcessBGEM3` (imports `FlagEmbedding`; used by ingestion on the VM) and `HttpEmbeddingClient` (calls the embedding service; used by the query API). Both return dense **and** sparse via a shared Pydantic contract (do not drop sparse, SPEC §7.2). Also stand up the **embedding service** FastAPI app on the VM (`GET /health`, `POST /embed`) loading the model once at startup.
5. `vector_store.py`: create the Qdrant collection (named dense + named sparse vectors), upsert chunks with full payload, dense search returning candidates + payload.
6. `synthesizer.py`: build a numbered context block, call Sarvam-105B with the SPEC §9 synthesis prompt (answer only from context; cite every claim; say "insufficient context" otherwise). Set `max_tokens` high enough for reasoning.
7. `provenance.py` (minimal): parse `[n]` markers → citation objects (SPEC §6.3) with `source_tier="internal"`; **discard/flag any uncited claim** (hard rule).
8. `scripts/run_ingestion.py`: the **real** ingestion entry point — CLI to ingest `data/raw/` → cloud Qdrant + Neo4j, **run on the GPU VM** with `InProcessBGEM3` (no HTTP hop for the batch job). It should write a machine-readable **run summary** (docs processed, chunks created, graph nodes/edges) that `POST /ingest` can later report. `api/routes.py`: `POST /query` embeds the question via `HttpEmbeddingClient` then calls synthesizer over dense-retrieved context; `POST /ingest` is a **thin admin-only status endpoint returning the last run summary** ([D-8](#11-deviations-log-keep-this-current)), not a job trigger. Keep business logic out of `routes.py` (SPEC §7.7).
9. Make `/health`'s cloud-Qdrant check real; add an embedding-service reachability check to `/health` (query-time embedding depends on it).

**✅ Tests (all vendors mocked):**
- Chunker: SOP splits by step, preserves order; prose falls back to fixed size.
- Embeddings wrapper returns both dense and sparse (fake model).
- Vector store upsert/search round-trip against an in-memory/fake Qdrant or a mocked client.
- Synthesizer: given fixed context + a fake LLM reply, produces answer + citations; **an uncited claim is dropped/flagged** (this is the SPEC §6.3 hard-rule test).
- `POST /query` happy path returns the SPEC §8 response shape.

**DoD:**
- Ingest a handful of real SOPs locally, ask 3–5 questions via `POST /query`, get grounded answers with `[n]` citations traceable to chunks. Unit suite green, lint + types clean. Committed. **This is the fallback demo; keep it working through every later phase.**

---

### Phase 2 — Hybrid search + Cohere rerank

**G:** Upgrade retrieval to dense + sparse fused with RRF (in Qdrant), then rerank the fused set with Cohere down to `RERANK_TOP_N`, with graceful fallback.

**T:**
1. `uv add cohere`. Re-verify Cohere v2 + Qdrant hybrid ([§3.2](#32-cohere-rerank), [§3.7](#37-qdrant-hybrid-search)).
2. `hybrid_search.py`: query Qdrant with dense + sparse prefetches + `FusionQuery(RRF)`; return ~20 candidates with payload (SPEC §7.3).
3. `reranker.py`: single reused `ClientV2`; batch all candidates in one `rerank` call (`model=RERANK_MODEL` default `rerank-v4.0-pro`, `top_n=RERANK_TOP_N`); attach Cohere relevance scores to each chunk for logging. Wrap with timeout + try/except; **on any failure, fall back to pre-rerank RRF order** (SPEC §7.3, §13).
4. Route `POST /query` through hybrid → rerank → synthesize.

**✅ Tests (mocked):**
- RRF fusion produces the expected merged ranking for known dense/sparse rank inputs.
- Reranker reorders per a fake Cohere response and keeps scores.
- **Reranker fallback:** on simulated Cohere timeout/error, retrieval still returns RRF-ordered results (no exception escapes). Required test.

**DoD:**
- Hybrid+rerank beats Phase-1 dense-only on a few keyword-heavy and semantic questions (spot-check). Fallback verified. Suite green, lint + types clean. Committed.

---

### Phase 3 — Knowledge graph + SOP versioning

**G:** Extract entities/relations into Neo4j per SPEC §6.2, add graph-aware retrieval, and implement the SOP `SUPERSEDES` version chain + `is_current_version` flagging.

**T:**
1. `uv add neo4j`. Re-verify driver 6.x ([§3.8](#38-neo4j-python-driver)).
2. `graph_store.py`: driver wrapper; `verify_connectivity` at startup; parameterized MERGE/MATCH; every node carries `chunk_ids` for citability (SPEC §6.2). Make `/health`'s Neo4j check real.
3. `graph_extractor.py`: LLM extraction of typed triples using the SPEC §9 graph-extraction prompt with the schema inlined (so the model can't invent relationship types). Validate output against SPEC §6.2 with Pydantic; reject/repair off-schema triples.
4. **Versioning:** on ingesting a new version of an existing SOP (title match), create `(:Process)-[:SUPERSEDES]->(:Process)` and set old chunks `is_current_version=false` (SPEC §7.1).
5. `graph_search.py`: extract candidate entity names from the question via an **LLM call** (Sarvam, `reasoning_effort=None` — [D-9](#11-deviations-log-keep-this-current)), fuzzy-match to node names, traverse 1–2 hops, return facts as text triples (SPEC §7.3).
6. Fuse graph facts with hybrid+reranked chunks into the synthesizer context. On graph failure, degrade to vector-only + log warning.

**✅ Tests (mocked LLM + fake/embedded Neo4j or mocked driver):**
- Extractor maps a fixed chunk to schema-valid triples; off-schema relations are rejected.
- Ingesting v2 of an SOP creates a `SUPERSEDES` edge and flips old chunks to `is_current_version=false`.
- Graph search: entity match → expected 1–2 hop facts for a seeded graph.
- Graph-failure degradation returns vector-only results without crashing.

**DoD:**
- ≥5 questions that require graph traversal (unanswerable by vector search alone) are answered correctly against a seeded corpus (SPEC §11). Versioning edge + flags verified. Suite green, lint + types clean. Committed.

---

### Phase 4 — Multimodal ingestion (tables + images/charts)

**G:** Extract and index tables (→ markdown) and images/charts (→ captions) so table/chart questions are answerable.

**T:**
1. Re-verify Docling multimodal ([§3.5](#35-docling-parsing--multimodal)). Captioning approach is **decided**: hosted Docling picture description ([D-3](#11-deviations-log-keep-this-current)). Runs on the VM as part of ingestion.
2. `parser.py`: also emit table blocks and image blocks with page numbers.
3. Tables → **markdown** (never raw HTML, SPEC §7.1) → chunk with `content_type="table"` → embed → Qdrant.
4. Images/charts → captions via **hosted Docling** ([§3.10](#310-gpu-vm-embedding-service--ingestion-host)): `enable_remote_services=True`, `do_picture_description=True`, `do_chart_extraction=True`, `PictureDescriptionApiOptions(url=DOCLING_VLM_URL, headers={Authorization}, params={"model": DOCLING_VLM_MODEL}, prompt=..., timeout=...)`. `multimodal.py` is a thin wrapper/orchestrator around this config. Store caption text (`content_type="image_caption"`) for retrieval **and** a reference to the original image file for optional UI display (SPEC §7.1). All VLM endpoint values come from config, not literals.
5. Ensure table/caption chunks flow through the full hybrid→graph→rerank→synthesize path.

**✅ Tests (mocked VLM/OCR):**
- Table item → markdown chunk with `content_type="table"`.
- Image item → `image_caption` chunk (fake captioner) + retained image-file reference.
- A table-derived chunk is retrievable and citable end-to-end.

**DoD:**
- ≥5 questions requiring table/chart content are answered correctly (SPEC §11). Suite green, lint + types clean. Committed.

---

### Phase 5 — Router agent + Exa web search + provenance

**G:** The core novelty: a tool-calling router (Sarvam-105B, reasoning disabled) that chooses `internal_search` / `web_search` / `answer_directly`, capped at `MAX_AGENT_ITERATIONS`, with full provenance tagging and version warnings — **plus multi-turn conversation memory** ([§4.9](#49-conversation-memory-multi-turn)).

**T:**
1. `uv add exa-py`. Re-verify Exa + Sarvam tool calling ([§3.1](#31-sarvam-llm-router--generation), [§3.3](#33-exa-web-search)).
2. `tools.py`: the three tool schemas (SPEC §7.4). `prompts.py`: router + synthesis + extraction + **follow-up-rewrite** prompts (SPEC §9 + [§4.9](#49-conversation-memory-multi-turn)), centralized.
3. `web_search.py`: Exa wrapper; every result carries `url` + `retrieved_at` (UTC ISO 8601); results held **only** in a per-request object, **never** written to Qdrant (SPEC §7.3).
4. `router.py`: tool-calling loop with `reasoning_effort=None`; encode the SPEC §7.4 routing **rules as guardrails** (default internal; web only when genuinely external or internal returns nothing above threshold); **cap at `MAX_AGENT_ITERATIONS`**, else return "couldn't find a confident answer" (no guessing); log every tool call + result.
5. **Conversation memory** ([§4.9](#49-conversation-memory-multi-turn), [D-10](#11-deviations-log-keep-this-current)): a `ConversationStore` behind an interface (in-memory for tests; lightweight persistent for real use). On each `POST /query`: load history by `conversation_id`, **rewrite follow-ups into a standalone query** (cheap Sarvam call) before routing/retrieval, include a bounded history window in synthesis context (conversational only, never a citation source), then append the new turn. Scope strictly by `conversation_id`; never cross-leak; never write history into Qdrant/Neo4j.
6. `provenance.py` (full): attach structured citation objects (SPEC §6.3), separate "From internal docs" vs "From the web", and **prepend a version warning** if any cited chunk is `is_current_version=false` (SPEC §7.5).
7. `POST /query` returns the full SPEC §8 response: `answer`, `citations`, `used_web_search`, `version_warning`, `tool_trace` (and echoes/creates `conversation_id`).

**✅ Tests (mocked LLM tool calls + mocked Exa):**
- Routing rules: SOP/product question → `internal_search`; explicitly-current-external question → `web_search`; greeting → `answer_directly`.
- Iteration cap: after `MAX_AGENT_ITERATIONS`, returns the low-confidence response, never fabricated content. Required test.
- Web results tagged `external` with `url` + `retrieved_at`, and **never persisted** to the vector store. Required test.
- Version warning fires when a superseded chunk is cited (SPEC §11 case). Required test.
- Uncited-claim hard rule still holds through the full pipeline.
- **Multi-turn:** a follow-up ("what about the previous version?") is rewritten to a standalone query using history and retrieves correctly. Required test.
- **Conversation isolation:** history from one `conversation_id` never appears in another; unknown id starts fresh. Required test.
- **Memory is not a citation source:** a claim supported only by prior-turn text (no retrieved chunk) is still dropped/flagged. Required test.

**DoD:**
- ≥2 questions correctly trigger web search and are labeled external; ≥1 superseded-SOP case surfaces the warning; a multi-turn follow-up is answered correctly using rewritten context; every golden answer has ≥1 citation or explicitly declines (SPEC §11). Suite green, lint + types clean. Committed.

---

### Phase 6 — Evaluation: RAGAS + ablation

**G:** A golden set + RAGAS metrics + the 4-stage ablation study, producing the table for the academic write-up.

**T:**
1. `uv add ragas`. **Pin the version** and pick the API (collections vs legacy) per [§3.9](#39-ragas); do not mix.
2. Build a golden set of **30–50** question / answer / ground-truth-context triples from the actual documents (SPEC §7.6). Ensure coverage of the SPEC §11 special cases: ≥5 graph-only, ≥5 table/chart, ≥2 web-search, ≥1 superseded-SOP.
3. `ragas_eval.py`: run **faithfulness, context precision, context recall, answer relevancy**. Configure the judge LLM explicitly (record which model — it affects scores).
4. `ablation.py`: run the golden set through **four configs** — naive (vector-only) → +hybrid+rerank → +graph → +graph+multimodal — recording metric deltas per stage (SPEC §7.6). Make the config switchable via flags/params, reusing the real pipeline components.
5. `scripts/run_eval.py`: CLI to run RAGAS + ablation and dump a report (JSON + a markdown table).

**✅ Tests (mocked judge LLM):**
- Golden-set loader validates structure and the required-case counts.
- Ablation runner executes all four configs and emits a complete metrics table (with a fake judge).
- Report writer produces valid JSON + markdown.

**DoD:**
- `run_eval.py` produces the ablation table across all four stages. If faithfulness does **not** monotonically improve, that's a **finding to document, not a bug to hide** (SPEC §11). Suite green, lint + types clean. Committed.

---

### Phase 7 — Hardening & demo readiness (backend side)

**G:** Make the backend robust and demo-ready. (Streamlit UI is frontend scope; backend must fully support SPEC §7.8's needs.)

**T:**
1. Confirm `POST /query` returns everything the UI needs: inline citation markers, internal-vs-external citation split, and `version_warning` (SPEC §7.8, §8).
2. Review timeouts, retries, and every degradation path (rerank/graph/web) under simulated failure. Confirm user-facing messages are clean and `request_id`-correlated.
3. Tighten logging: token usage + tool traces present and useful; no secrets/PII/full bodies in logs.
4. Load a representative corpus; sanity-check latency and Qdrant/Neo4j free-tier limits (SPEC §13). Document any limits hit.
5. Finalize `README.md` (human quickstart: env setup, `uv sync`, ingestion, run server, run eval) and refresh `.env.example`.
6. Ensure the full test suite (unit) is green and the optional integration suite is documented as opt-in.

**✅ Tests:**
- End-to-end (mocked vendors) covering: internal-only, hybrid, graph, multimodal, web-routed, superseded-SOP, and **multi-turn follow-up** flows each return the correct SPEC §8 shape.
- Degradation matrix: each dependency failing individually still yields a valid, correctly-labeled response.

**DoD:**
- All SPEC §11 success criteria demonstrably met. Every dependency-down path degrades gracefully. Docs current. Full suite + lint + types green. Tagged/committed as the demo build.

---

## 6. Cross-Phase Definition of Done (the bar for "done")

A phase is complete only when **all** hold:

- [ ] All phase tasks implemented against the current (re-verified) APIs.
- [ ] Unit tests for every new `src/` module, all vendors mocked, all green in the default `uv run pytest`.
- [ ] `uv run ruff check` and `uv run ruff format --check` clean.
- [ ] `uv run pyrefly check` clean.
- [ ] No hardcoded secrets/paths/model names; new env keys added to `.env.example`.
- [ ] Every new external call has timeout + error handling + (where relevant) graceful degradation.
- [ ] Structured logs present for new tool/retrieval/LLM calls; no secrets leaked.
- [ ] The phase's concrete DoD criteria are manually verified (not just "tests passed").
- [ ] Committed with a clear message; `uv.lock` updated; pre-commit green.

---

## 7. Testing Conventions

- **Layout:** tests in `backend/tests/` mirroring `src/` (SPEC §3 already lists the required files: `test_chunker.py`, `test_hybrid_search.py`, `test_graph_search.py`, `test_router.py`, `test_citation_grounding.py`). Add more as modules grow.
- **Frameworks:** `pytest` + `pytest-asyncio` for async paths.
- **Isolation:** unit tests mock **all** network/vendor calls. Provide reusable fixtures/fakes: a fake embedder (returns deterministic dense+sparse), a fake LLM (canned tool-calls + completions), a fake reranker, a fake Exa, and a fake/embedded Qdrant + Neo4j.
- **Markers:** `@pytest.mark.integration` for anything hitting live services; excluded from the default run via `addopts = ["-m", "not integration"]`. Document how to run them opt-in in the README.
- **Cost guard:** never call Cohere/Exa/Sarvam in a loop in tests (SPEC §13). If you must smoke-test a live call, gate it behind an integration marker and a single invocation.
- **Determinism:** no reliance on wall-clock, network, or model nondeterminism in unit tests. Freeze time where `retrieved_at`/timestamps are asserted.
- **The behavior tests that must always exist and pass** (they encode SPEC's non-negotiables + the multi-turn decision):
  1. Uncited claim is dropped/flagged (never surfaced) — and prior-turn history does **not** count as a citation.
  2. Router respects `MAX_AGENT_ITERATIONS` and declines rather than guesses.
  3. Web results are labeled `external` and never persisted to Qdrant.
  4. Reranker failure falls back to RRF order.
  5. Superseded SOP triggers a version warning.
  6. Multi-turn: follow-ups are rewritten to standalone queries; conversation history is isolated by `conversation_id` and never leaks or gets written to the retrieval stores.

---

## 8. Suggested Dependencies (verify versions with `uv add`)

Split into dependency groups so the CPU-only API image never pulls GPU deps ([§2.2](#22-deployment-topology-decided--implement-to-this-shape)):

- **API/runtime (default):** `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `structlog`, `tenacity`, `httpx`, `qdrant-client`, `neo4j`, `cohere`, `exa-py`, `sarvamai` (or `openai` pointed at Sarvam), `ragas`.
- **`gpu` group (VM only — `uv add --group gpu`):** `docling`, `FlagEmbedding`, `torch`. The embedding-service app reuses `fastapi`/`uvicorn` from runtime plus this group.
- **Dev (`--dev`):** `pytest`, `pytest-asyncio`, `ruff`, `pyrefly`, `pre-commit`.

> Let `uv add` resolve current versions; don't hand-pin unless a constraint is needed. Pin **RAGAS** (eval stability) and any SDK where a breaking change would silently skew results. `torch` must be the CUDA build on the VM (L40S). Prefer well-known packages; flag anything that looks like a typosquat.

---

## 9. Definitions (shared vocabulary)

- **Chunk** — a unit stored in Qdrant with the SPEC §6.1 payload; `content_type ∈ {text, table, image_caption}`.
- **Trust tier / provenance** — `internal` (ingested docs) vs `external` (live web); external is set only at query time and never persisted (SPEC §6.1, §7.3).
- **RRF** — Reciprocal Rank Fusion; how dense + sparse candidate lists are merged (in Qdrant).
- **Graceful degradation** — on a dependency failure, return a lower-quality but valid, correctly-labeled answer instead of erroring.
- **Version warning** — a notice prepended to an answer when a cited SOP chunk is `is_current_version=false`.
- **Golden set** — 30–50 curated Q/A/ground-truth-context triples used for RAGAS + ablation.
- **Conversation memory** — per-`conversation_id` turn history used for follow-up rewriting and conversational context ([§4.9](#49-conversation-memory-multi-turn)); never a citation source and never written to Qdrant/Neo4j.
- **Follow-up contextualization** — rewriting a terse follow-up into a standalone query using recent history before retrieval/routing.
- **DoD** — Definition of Done; the checklist that gates moving to the next phase.

---

## 10. Anti-patterns (do not do these)

- Using `sarvam-30b`/`sarvam-m` (deprecated) or `rerank-v3.5` as the default (superseded by `rerank-v4.0-pro`).
- Assuming Sarvam Vision can caption arbitrary charts/images ([§3.4](#34-sarvam-vision--important-architecture-correction)) — it's OCR/Doc-AI.
- `pip`, `requirements.txt`, or hand-editing dependency versions in `pyproject.toml`.
- Persisting Exa web results into the internal chunk store.
- Letting a raw vendor exception reach the API; letting a reranker/graph/web outage crash a request.
- Emitting an uncited claim, or letting the router guess after the iteration cap.
- Hardcoding keys, URLs, paths, or model names.
- Calling billed/rate-limited APIs (Cohere/Exa) in test loops.
- Business logic inside `routes.py` (keep it in `agent/`, `retrieval/`, `generation/`).
- Importing `torch`/`FlagEmbedding`/`docling` into the query API's dependency closure — those are GPU-VM/`gpu`-group only ([§2.2](#22-deployment-topology-decided--implement-to-this-shape)).
- Exposing the VM embedding service to the public internet (put it behind a private network/token).
- Skipping the ablation study because it isn't user-facing (SPEC §7.6).

---

## 11. Deviations Log (keep this current)

Record any place the implementation departs from SPEC.md, with the reason and date. Start with the ones already known:

| # | Deviation from SPEC | Reason | Date |
|---|---|---|---|
| D-1 | Use `sarvam-105b` only (SPEC lists `sarvam-30b` in tables) | `sarvam-30b`/`sarvam-m` deprecated by Sarvam | 2026-09-13 |
| D-2 | Reranking via **OpenRouter `/rerank`** (model `cohere/rerank-4-fast` default; `cohere/rerank-4-pro` for eval), not Cohere's native `cohere` SDK / `/v2/rerank`. SPEC says `rerank-v3.5`. `COHERE_API_KEY` removed; auth via `OPENROUTER_API_KEY`. | v4 is current SOTA and `fast` keeps the per-query rerank hop low-latency; routing rerank through OpenRouter unifies the API/billing layer. Kept configurable via `RERANK_MODEL`. | 2026-09-14 |
| D-3 | Image/chart captioning via **hosted Docling** `do_picture_description`/`do_chart_extraction` (OpenAI-compatible VLM), not a bespoke Sarvam Vision VLM call | Sarvam Vision is OCR/Doc-AI, not a captioning VLM; user chose hosted Docling | 2026-09-13 |
| D-4 | BGE-M3 + Docling run on a **GPU VM**; query API embeds via an **HTTP embedding service**; ingestion runs **on the VM** in-process (SPEC implies a single service) | User's chosen topology (L40S VM); keeps API CPU-only and GPU work co-located | 2026-09-13 |
| D-5 | Type checker is **Pyrefly** (SPEC §12 says type hints, unspecified checker) | User preference | 2026-09-13 |
| D-6 | New config keys `EMBEDDING_SERVICE_URL/TOKEN`, `DOCLING_VLM_URL/TOKEN/MODEL` added beyond SPEC §5 | Required by the VM topology and hosted captioning | 2026-09-13 |
| D-7 | Self-hosted VLM for captioning = **vLLM + Qwen2.5-VL-7B-Instruct** on the L40S (OpenAI-compatible) | User chose a self-hosted VLM endpoint; 7B fits single-GPU alongside BGE-M3 | 2026-09-13 |
| D-8 | `POST /ingest` is a thin admin **status** endpoint (returns last ingestion summary); real ingestion is `run_ingestion.py` on the VM | Ingestion needs the GPU; avoids cross-box job orchestration; matches SPEC §1.4 batch-script intent | 2026-09-13 |
| D-9 | Graph-search entity extraction is **LLM-based** (Sarvam, reasoning disabled) | User choice; higher recall, Sarvam already integrated | 2026-09-13 |
| D-10 | **Multi-turn conversation memory** via `conversation_id` (storage + history-aware retrieval + follow-up contextualization) — beyond SPEC's single-turn framing | User choice; see [§4.9](#49-conversation-memory-multi-turn) | 2026-09-13 |
| _add rows as you discover more_ | | | |

---

## 12. Open Questions (ask, do NOT assume)

Resolve these with the user before the phase that depends on them. Do not guess.

**Resolved (moved to decisions/deviations):** Q-1 captioning → hosted Docling ([D-3](#11-deviations-log-keep-this-current)); Q-2 judge → Sarvam via LiteLLM ([§3.9](#39-ragas)); Q-3 type checker → Pyrefly ([D-5](#11-deviations-log-keep-this-current)); Q-4 stores → Qdrant Cloud + Neo4j AuraDB ([§2.2](#22-deployment-topology-decided--implement-to-this-shape)); Q-5 embeddings → L40S GPU VM, HTTP service + in-process ingestion ([D-4](#11-deviations-log-keep-this-current), [§3.10](#310-gpu-vm-embedding-service--ingestion-host)); Q-A/Q-B/Q-C confirmed.

**All resolved (kept here for provenance; decisions live in the phases + deviations log):**

- ~~Q-6~~ **Resolved:** graph-search entity extraction is **LLM-based** (a Sarvam call with `reasoning_effort=None`), not local NER — Sarvam is already in the loop and recall matters more than the extra call. See [D-9](#11-deviations-log-keep-this-current); implemented in Phase 3.
- ~~Q-7~~ **Resolved:** `conversation_id` drives **real multi-turn memory**. This adds conversation storage, history-aware query handling, and follow-up contextualization. See [D-10](#11-deviations-log-keep-this-current) and [§4.9 Conversation memory](#49-conversation-memory-multi-turn); implemented in Phase 5.
- ~~Q-8~~ **Resolved:** ingestion's real entry point is `run_ingestion.py` on the VM; `POST /ingest` is a **thin admin-only endpoint that returns the last ingestion summary**, not a job trigger. Matches SPEC §1.4. See [D-8](#11-deviations-log-keep-this-current).

> No open questions remain. If new ambiguity appears during implementation, add it here and ask before guessing.
