# SPEC.md — Graph-Augmented Multimodal Company-Knowledge QA System

> **Read this entire file before writing any code.** This is the single source of truth for architecture, schemas, and behavior. If something here conflicts with a general best-practice you know, follow this spec — it encodes project-specific decisions already made. If something here is ambiguous or missing, stop and ask rather than guessing.
>
> **API surfaces evolve.** Sarvam, Cohere, Exa, Qdrant, and Neo4j client libraries may have changed method names/params since this spec was written. Treat every code snippet below as *intent*, not verbatim API truth — check the live SDK/docs for exact signatures before finalizing integration code.

---

## 1. Project Overview

### 1.1 One-line pitch
A Question-Answering system over internal company documents (SOPs, product info) that combines **hybrid vector search**, a **domain-specific knowledge graph**, **multimodal document understanding** (tables/charts/images), and **routed live web search**, with every answer citation-grounded and source-tagged by trust tier.

### 1.2 Why this exists (grading/demo narrative — keep this in mind for every feature)
Standard RAG (embed → retrieve → generate) fails company-knowledge use cases in three specific ways this project fixes:
1. It discards tables, charts, and diagrams during text extraction — this project extracts and indexes them.
2. It can't answer relationship questions ("which SOP governs system X", "is this the current SOP version") — this project adds a knowledge graph for exactly that.
3. It can't distinguish "verified internal policy" from "something an LLM inferred or found on the open web" — this project tags every citation with a trust tier.

### 1.3 Goals
- Answer questions about internal SOPs and product docs, grounded only in retrieved evidence.
- Answer relational/multi-hop questions using the knowledge graph, not just similarity search.
- Correctly extract and use information from tables, charts, and images in source documents.
- Route to live web search only when the question genuinely requires current external information, and clearly label that content as external.
- Detect and surface SOP version conflicts (e.g., warn if a superseded SOP would otherwise be cited).
- Produce every answer with inline citations traceable to a specific chunk or graph fact.
- Produce a quantitative ablation report (naive → hybrid → +graph → +multimodal) suitable for an academic evaluation section.

### 1.4 Non-goals (explicitly out of scope for this build)
- User authentication / role-based document access control (mention as future work only).
- Multi-tenant support.
- Fine-tuning any model.
- Production-grade horizontal scaling, load balancing, or high availability.
- Voice input/output (Sarvam has speech APIs; not part of this project unless separately requested).
- Real-time document sync / webhook ingestion — ingestion is a manual/batch script for this project.

---

## 2. High-Level Architecture

```
                        [ User Question ]
                                │
                    [ Router Agent — Sarvam-105B, tool-calling ]
                                │
        decides which tool(s) to call, max 3 iterations
        ┌───────────────┬──────────────────┬─────────────────┐
        │               │                  │
[ Internal Vector    [ Internal Graph   [ Live Web Search
  Search (Qdrant) ]    Traversal (Neo4j)]  (Exa API) ]
        │               │                  │
   SOP/product chunks  Process→Step→Role   Fresh external
   + table/image       →System edges;      info, full page
   captions             SOP version edges   content
        └───────────────┴──────────────────┘
                        │
              Fuse + rerank (Cohere Rerank API)
           + tag provenance: internal vs external
                        │
          [ Sarvam-105B generates answer with
            per-claim citations + source tier ]
                        │
                  [ Answer to user, FastAPI → frontend ]
```

### 2.1 Ingestion pipeline (offline / batch, separate from query-time flow)

```
Raw docs (PDF/DOCX) → Docling parse → split into:
  ├─ narrative text  → chunk → embed (BGE-M3) → Qdrant
  ├─ tables          → markdown → embed (BGE-M3) → Qdrant
  └─ images/charts   → Sarvam Vision caption → embed (BGE-M3) → Qdrant
                     ↘
                  entity/relation extraction (LLM) → Neo4j graph
```

---

## 3. Repository Structure

Create exactly this layout. Do not invent alternative top-level folders without updating this spec first. The repo is split into a Python `backend/` and a `frontend/`, with shared docs under `docs/`.

```
/
├── docs/
│   └── SPEC.md                      # this file
├── README.md                        # human-facing quickstart
├── backend/
│   ├── pyproject.toml               # uv-managed project + dependencies
│   ├── uv.lock                      # uv lockfile (committed)
│   ├── .python-version              # pins interpreter for uv
│   ├── .env.example                 # documented, no real secrets
│   ├── data/
│   │   ├── raw/                     # source SOPs/product docs (gitignored)
│   │   └── processed/               # intermediate parsed output (gitignored)
│   ├── src/
│   │   ├── ingestion/
│   │   │   ├── parser.py            # Docling wrapper: text/table/image split
│   │   │   ├── chunker.py           # SOP-aware step-based chunking
│   │   │   ├── multimodal.py        # Sarvam Vision captioning calls
│   │   │   └── graph_extractor.py   # LLM-based entity/relation extraction
│   │   ├── indexing/
│   │   │   ├── vector_store.py      # Qdrant client wrapper
│   │   │   ├── graph_store.py       # Neo4j client wrapper
│   │   │   └── embeddings.py        # BGE-M3 embedding wrapper
│   │   ├── retrieval/
│   │   │   ├── hybrid_search.py     # dense + BM25 + RRF fusion
│   │   │   ├── graph_search.py      # entity match → 1-2 hop traversal
│   │   │   ├── reranker.py          # Cohere Rerank API wrapper
│   │   │   └── web_search.py        # Exa API wrapper
│   │   ├── agent/
│   │   │   ├── router.py            # tool-calling router logic (Sarvam-105B)
│   │   │   ├── tools.py             # tool schemas exposed to the router
│   │   │   └── prompts.py           # all system/user prompt templates
│   │   ├── generation/
│   │   │   ├── synthesizer.py       # final answer generation + citation grounding
│   │   │   └── provenance.py        # source-tier tagging logic
│   │   ├── evaluation/
│   │   │   ├── ragas_eval.py        # faithfulness / context precision / recall
│   │   │   └── ablation.py          # runs the 4-stage ablation study
│   │   └── api/
│   │       ├── main.py              # FastAPI app
│   │       ├── routes.py
│   │       └── schemas.py           # Pydantic request/response models
│   ├── scripts/
│   │   ├── run_ingestion.py         # CLI: ingest data/raw → Qdrant + Neo4j
│   │   └── run_eval.py              # CLI: run RAGAS + ablation, dump report
│   └── tests/
│       ├── test_chunker.py
│       ├── test_hybrid_search.py
│       ├── test_graph_search.py
│       ├── test_router.py
│       └── test_citation_grounding.py
└── frontend/
    └── app.py                       # Streamlit chat UI
```

> All `uv` commands (`uv sync`, `uv run ...`) are run from inside `backend/`. Paths like `data/raw/` and `src/...` in this spec are relative to `backend/` unless stated otherwise.

---

## 4. Tech Stack

| Layer | Choice | Package/Service | Notes |
|---|---|---|---|
| Doc parsing | Docling | `docling` | Handles PDF/DOCX layout, tables, embedded images |
| Embeddings | BGE-M3 | `FlagEmbedding` or `sentence-transformers` | Multilingual, supports dense+sparse in one model |
| Vector store | Qdrant | `qdrant-client` (local Docker or Qdrant Cloud free tier) | Native hybrid search support |
| Graph store | Neo4j | `neo4j` Python driver (AuraDB free tier) | No server management needed |
| Reranker | Cohere Rerank | `cohere` SDK (`rerank-v3.5` or current model) | Hosted rerank API; needs `COHERE_API_KEY`, watch free-tier rate limits |
| LLM (router + generation) | Sarvam-105B | `sarvamai` SDK, or OpenAI SDK pointed at `https://api.sarvam.ai/v1` | OpenAI-compatible endpoint; supports tool calling |
| Vision / doc intelligence | Sarvam Vision | Sarvam API | For table/chart/image captioning |
| Live web search | Exa | `exa-py` | Use `auto` or `fast` search type for latency |
| Orchestration | LlamaIndex (`PropertyGraphIndex`) or hand-rolled | `llama-index`, `llama-index-graph-stores-neo4j` | Optional — hand-rolled retrieval is also acceptable and gives more control over the hybrid+graph fusion logic |
| Backend | FastAPI | `fastapi`, `uvicorn` | |
| Frontend | Streamlit | `streamlit` | Fastest path to a working demo |
| Evaluation | RAGAS | `ragas` | Faithfulness, context precision/recall, answer relevancy |
| Python packaging | uv | `uv` | Manages venv, deps, and lockfile for `backend/`; no `pip`/`requirements.txt` |

**Rule for coding agents:** before wiring up any of the above, fetch that library's current docs/README rather than relying on memorized method signatures. API surfaces (especially Sarvam's and Cohere's, since they're fast-moving products) change.

**Dependency management (uv):** the Python backend uses [`uv`](https://docs.astral.sh/uv/), not pip. All dependencies live in `backend/pyproject.toml` with a committed `backend/uv.lock`. Use:
- `uv add <package>` to add a dependency (do not hand-edit `pyproject.toml` for versions).
- `uv sync` to install/refresh the environment from the lockfile.
- `uv run <cmd>` to run commands inside the project environment (e.g. `uv run uvicorn src.api.main:app`, `uv run python scripts/run_ingestion.py`, `uv run pytest`).
Do not create `requirements.txt` or use `pip install` in this project.

---

## 5. Environment Variables (`.env`)

```
SARVAM_API_KEY=
COHERE_API_KEY=                     # for the Cohere Rerank API
EXA_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
NEO4J_URI=
NEO4J_USER=
NEO4J_PASSWORD=
EMBEDDING_MODEL=BAAI/bge-m3
RERANK_MODEL=rerank-v3.5            # Cohere rerank model; verify current model name against Cohere docs
RERANK_TOP_N=5                     # candidates returned after rerank
LLM_MODEL=sarvam-105b
LLM_ROUTER_REASONING_EFFORT=null    # disable "thinking mode" for router/classification calls to save cost+latency
LLM_SYNTH_REASONING_EFFORT=low      # allow reasoning for final answer synthesis
MAX_AGENT_ITERATIONS=3
```

`.env` / `.env.example` live in `backend/`. Never commit `.env`. `.env.example` should list every key above with an empty value and a one-line comment.

---

## 6. Data Schemas

These schemas are contracts between components. Do not change field names without updating every consumer.

### 6.1 Chunk schema (stored in Qdrant payload)

```json
{
  "chunk_id": "uuid",
  "doc_id": "uuid",
  "doc_title": "string",
  "doc_type": "SOP | product_info",
  "content_type": "text | table | image_caption",
  "text": "string — the actual chunk content (markdown for tables)",
  "page_number": "int | null",
  "sop_version": "string | null",       // e.g. "v2.1" — null for product_info
  "effective_date": "ISO 8601 date | null",
  "is_current_version": "bool",         // false if a newer version exists
  "entity_ids": ["graph node IDs mentioned in this chunk"],
  "source_tier": "internal"             // always "internal" for ingested docs; "external" is set only on live web-search results at query time, never stored
}
```

### 6.2 Knowledge graph schema (Neo4j)

**Node labels and key properties:**

| Label | Key properties |
|---|---|
| `Process` | name, doc_id, sop_version |
| `Step` | name, order, process_id |
| `Role` | name (e.g. "Sales Associate", "IT Admin") |
| `System` | name (e.g. "SAP", "internal ticketing tool") |
| `ComplianceRequirement` | name, regulation_ref |
| `Product` | name, sku |
| `Feature` | name |
| `Spec` | name, value, unit |

**Relationship types:**

```
(Process)-[:HAS_STEP]->(Step)
(Step)-[:OWNED_BY]->(Role)
(Step)-[:REQUIRES]->(System)
(Process)-[:MUST_SATISFY]->(ComplianceRequirement)
(Process)-[:SUPERSEDES]->(Process)          # version chain — critical for the "outdated SOP" novelty feature
(Product)-[:HAS_FEATURE]->(Feature)
(Product)-[:HAS_SPEC]->(Spec)
(Product)-[:COMPATIBLE_WITH]->(Product)
```

Every node also carries `chunk_ids: [str]` linking back to the source chunk(s) it was extracted from, so graph facts remain citable.

### 6.3 Citation object (attached to every generated claim)

```json
{
  "claim_text": "string",
  "source_tier": "internal | external",
  "reference": {
    "chunk_id": "uuid",            // if internal
    "doc_title": "string",
    "sop_version": "string | null",
    "is_current_version": "bool",
    "url": "string",               // if external (from Exa)
    "retrieved_at": "ISO 8601 timestamp"  // if external
  }
}
```

**Hard rule:** the generation module must refuse to emit a claim with no citation object. If the LLM produces an uncited sentence, either discard it or flag it for review — never present it as-is.

---

## 7. Component Specifications

### 7.1 Ingestion — `src/ingestion/`

- **parser.py**: Docling parses each doc into narrative text blocks, table blocks, and image blocks, preserving page numbers.
- **chunker.py**: For `doc_type == "SOP"`, chunk **by step**, not by fixed token count — each `Step` node's text should map to one chunk. Preserve step order as metadata. For `product_info`, chunk by section/feature. Fixed-size chunking is only a fallback for unstructured prose sections.
- **multimodal.py**: Every extracted table is converted to markdown before embedding (do not embed raw table HTML). Every extracted image/chart is sent to Sarvam Vision for a caption; store both the caption text (for embedding/retrieval) and a reference to the original image file (for optional display in the UI).
- **graph_extractor.py**: LLM-based extraction of (entity, relationship, entity) triples per chunk, mapped to the schema in §6.2. When ingesting a new version of an existing SOP, detect the prior version by doc title match and create a `SUPERSEDES` edge; mark the old version's chunks `is_current_version: false`.

### 7.2 Indexing — `src/indexing/`
Thin wrappers around Qdrant and Neo4j clients. `embeddings.py` wraps BGE-M3 and exposes both dense and sparse vectors for hybrid search — don't silently drop the sparse output.

### 7.3 Retrieval — `src/retrieval/`
- **hybrid_search.py**: Dense + BM25/sparse, fused with Reciprocal Rank Fusion (RRF). Return top ~20 candidates before reranking.
- **graph_search.py**: Extract candidate entity names from the question (simple NER or LLM call), match against graph node names (fuzzy match acceptable), traverse 1–2 hops, return connected facts as text triples.
- **reranker.py**: Calls the Cohere Rerank API (`RERANK_MODEL`) to rerank the fused candidate set down to top `RERANK_TOP_N` (~5). Send the query plus the candidate documents' text and keep Cohere's returned relevance scores on each chunk for logging/debugging. Wrap the call with try/except + timeout; on API failure, fall back to the pre-rerank RRF order (degrade gracefully rather than dropping results). Since this is a hosted, billed call, batch candidates in a single request and respect free-tier rate limits — don't call it in a tight loop during tests.
- **web_search.py**: Exa wrapper. Always attach `retrieved_at` timestamp and source URL to every result. Never merge Exa results into the "internal" chunk store — keep them in a separate in-memory object for the current request only.

### 7.4 Agent / Router — `src/agent/`
This is the core novelty component. The router is a tool-calling LLM call (Sarvam-105B, reasoning disabled for speed) that receives the question and three tool definitions:

```
tools = [
  { "name": "internal_search", "description": "Search internal SOPs and product docs (vector + graph)" },
  { "name": "web_search", "description": "Search the live web for current external information" },
  { "name": "answer_directly", "description": "Use only if the question is a greeting/meta-question needing no retrieval" }
]
```

**Routing rules (encode these explicitly, don't leave it purely to the model's judgment without guardrails):**
1. Default to `internal_search` for anything about SOPs, processes, products, or "how do we / what is our policy on X".
2. Only call `web_search` if the question explicitly needs current external info (regulations, market data, competitor info) or if `internal_search` returns no relevant results above a similarity threshold.
3. Cap total tool calls per question at `MAX_AGENT_ITERATIONS` (default 3). If still insufficient after the cap, return a "couldn't find a confident answer" response rather than guessing.
4. Log every tool call and its result for later evaluation/debugging.

### 7.5 Generation — `src/generation/`
- **synthesizer.py**: Takes fused, reranked context (internal chunks + graph facts + any web results) and generates the answer. System prompt (see §9) must require per-sentence citation tagging.
- **provenance.py**: Post-processes the raw LLM output to attach structured citation objects (§6.3) and to visually/textually separate "From internal docs" vs "From the web" in the final response. Also: if any cited chunk has `is_current_version: false`, prepend a warning to the answer (e.g., "Note: the SOP referenced here has since been updated to v3.0").

### 7.6 Evaluation — `src/evaluation/`
- **ragas_eval.py**: Build a golden set of 30–50 question/answer/ground-truth-context triples from your actual documents. Run RAGAS faithfulness, context precision, context recall, and answer relevancy.
- **ablation.py**: Run the same golden set through four configurations — naive (vector-only), +hybrid+rerank, +graph, +graph+multimodal — and record the metric deltas. This produces the table for the academic report; do not skip this even though it's not user-facing.

### 7.7 API — `src/api/`
Minimal REST surface (see §8 for exact contract). Keep business logic out of `routes.py` — it should only call into `agent/router.py` and return the result.

### 7.8 Frontend — `frontend/app.py`
Streamlit chat interface. Display: the answer text, inline citation markers, an expandable "Sources" section separating Internal vs External citations, and a version-warning banner when applicable.

---

## 8. API Contract

### `POST /query`
Request:
```json
{ "question": "string", "conversation_id": "string | null" }
```
Response:
```json
{
  "answer": "string",
  "citations": [ /* array of citation objects, see §6.3 */ ],
  "used_web_search": "bool",
  "version_warning": "string | null",
  "tool_trace": [ "internal_search", "web_search" ]
}
```

### `POST /ingest` (admin/dev use, not for end users)
Triggers `scripts/run_ingestion.py` against files in `data/raw/`. Returns a job summary (docs processed, chunks created, graph nodes/edges created).

### `GET /health`
Basic liveness check — confirms Qdrant and Neo4j connections are alive.

---

## 9. Prompt Templates (starting point — iterate as needed)

**Router system prompt:**
```
You are a routing agent for a company knowledge assistant. Given a user question,
decide which tool(s) to call. Default to internal_search for anything about company
SOPs, processes, or products. Only use web_search for questions that explicitly need
current external information, or if internal_search returns nothing relevant.
Never answer from your own memory about company-specific facts — always retrieve first.
```

**Synthesis system prompt:**
```
Answer the user's question using ONLY the provided context. For every factual claim,
add a citation marker like [1], [2] referencing the numbered source list provided.
If a claim cannot be supported by the provided context, do not include it.
If the context is insufficient to answer, say so explicitly rather than guessing.
If any source is marked as a superseded/outdated version, mention that explicitly.
```

**Graph extraction prompt:** instruct the LLM to output strict JSON triples matching §6.2's node/edge types — provide the schema in the prompt itself so the model doesn't invent new relationship types.

---

## 10. Build Order (do not skip ahead — each phase should be demoable before starting the next)

1. **Phase 0**: Repo scaffold, `.env.example`, health-check endpoint.
2. **Phase 1**: Naive text-only RAG — parse a few SOPs, chunk, embed, vector search, Sarvam-105B answers with basic citations. This is your safety net; get it fully working end-to-end first.
3. **Phase 2**: Add hybrid search (BM25 + dense + RRF) and Cohere Rerank reranking.
4. **Phase 3**: Add the knowledge graph — entity/relation extraction, Neo4j storage, graph-aware retrieval, SOP versioning edges.
5. **Phase 4**: Add multimodal ingestion — table-to-markdown, Sarvam Vision image captioning.
6. **Phase 5**: Add the router agent and Exa web search tool, with provenance tagging.
7. **Phase 6**: Run RAGAS + ablation study across all stages; write up the results.
8. **Phase 7**: Frontend polish, version-warning UI, final demo prep.

At the end of each phase, run the relevant tests in `tests/` before moving on.

---

## 11. Evaluation & Success Criteria

- RAGAS faithfulness score should improve at each ablation stage (naive → hybrid → graph → multimodal); if it doesn't, that's a real finding to discuss, not a bug to hide.
- At least 5 test questions must specifically require graph traversal (unanswerable from vector search alone) and be answered correctly.
- At least 5 test questions must require table/chart content and be answered correctly.
- At least 2 test questions must trigger the web-search tool and be correctly labeled as external.
- At least 1 test case must involve a superseded SOP version and correctly surface the version warning.
- Every answer in the golden test set must contain at least one citation, or explicitly state it cannot answer.

---

## 12. Coding Standards

- Python 3.10+, type hints on all function signatures. Backend deps managed with `uv` (see §4) — no `pip`, no `requirements.txt`.
- Pydantic models for all API request/response and inter-component data (don't pass raw dicts across module boundaries).
- Every external API call (Sarvam, Cohere, Exa, Qdrant, Neo4j) wrapped with try/except and sensible timeouts — these are network calls and will fail sometimes. The reranker in particular must degrade gracefully to RRF order on Cohere failure.
- Log tool calls, retrieval results, and token usage — you'll need this for debugging and for the evaluation writeup.
- No hardcoded API keys or file paths — everything through `.env` / config.
- Write a unit test alongside each new module in `src/`, not just at the end.

---

## 13. Known Constraints / Gotchas

- Sarvam-105B has "thinking mode" on by default, which consumes `max_tokens` on reasoning before producing content — set `reasoning_effort: null` for the router and any latency-sensitive call, and ensure `max_tokens` is high enough on synthesis calls that use reasoning, or you'll get truncated/empty responses.
- Sarvam's knowledge cutoff is mid-2025 with some reported hallucination on ungrounded queries — this is exactly why citation grounding (§6.3, §9) is non-negotiable, not a nice-to-have.
- Cohere Rerank is a hosted, rate-limited/billed API — confirm the current model name (`rerank-v3.5` may have moved) and free-tier limits before wiring it in, and reuse a single client. Because reranking runs on every query, a Cohere outage or rate-limit must not break retrieval: fall back to RRF order.
- Exa's free tier has a monthly credit cap — budget your web-search test calls accordingly during development; don't call it in a tight test loop.
- Qdrant and Neo4j free/cloud tiers may have storage or connection limits — check current limits before ingesting your full corpus.
- Confirm exact Sarvam Vision request/response format against current docs before building `multimodal.py` — this spec describes intent, not a verified API contract.
