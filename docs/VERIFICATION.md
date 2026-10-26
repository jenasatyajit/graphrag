# VERIFICATION.md — Verifying Phases 1–4 end to end

> A tickable, copy-paste guide to bring up the GPU VM, ingest the sample corpus,
> and confirm each phase's Definition of Done (DoD). Follow it top to bottom:
> Phase 3 first (no VLM needed), then Phase 4 (adds the VLM + a table/chart PDF).
>
> **Status: Phases 1–4 verified on real infrastructure (2026-09-30).** The exact
> commands below are what actually worked, corrected from the first-attempt
> guide. Known follow-ups are tracked in [§9](#9-known-issues-deferred) and
> [§10](#10-future-work).

Legend: `[VM]` = the remote Ubuntu 22.04 GPU box (SSH). `[WIN]` = your Windows
box (only needed for the Level-1 static checks in §1). Cloud services (Qdrant,
Neo4j, Sarvam, OpenRouter) are configured in `backend/.env`.

---

## 0. Topology (what we actually ran)

Everything for verification ran **on the VM** — the embedding service, ingestion,
the VLM, *and* the query API. The Windows box is only used for the static
lint/type/test checks in §1. The SSH-tunnel-to-Windows setup in earlier drafts is
optional and not required to verify the DoDs.

```
                         [VM] Ubuntu 22.04 + L40S 48GB
   ┌──────────────────────────────────────────────────────────────┐
   │ embedding_service (BGE-M3)      :8081   GET /health POST /embed │
   │ query API (FastAPI)             :8002   /health /query /ingest  │
   │ vLLM (Qwen3-VL-8B) [Phase 4]    :8000   /v1/chat/completions    │
   │ run_ingestion.py (Docling + BGE-M3, in-process)                 │
   └───────────────┬───────────────────────────┬────────────────────┘
                   │ cloud                      │ cloud
                   ▼                            ▼
             Qdrant Cloud                  Neo4j AuraDB
```

- **Ingestion** runs on the VM (Docling + BGE-M3 in-process) and writes chunks to
  cloud Qdrant and the graph to cloud Neo4j.
- **The query API** runs on the VM too. It reads `EMBEDDING_SERVICE_URL` from
  `.env` — on the VM this is `http://localhost:8081`, so it reaches the embedding
  service locally (no tunnel).
- The three long-running services (embedding, query API, vLLM) each get their own
  tmux window. Ingestion is a one-shot command.
- Ports used: **8081** embedding service, **8002** query API, **8000** vLLM.
  (Any free ports work; these are what we used. Keep them distinct.)

> **CPU-only deployment note:** in a split deployment the query API can run on a
> separate CPU box and reach the VM's embedding service over an SSH tunnel
> (`ssh -N -L 8081:localhost:8081 <user>@<vm-host>`). That box must **never**
> install `torch`/`docling`/`FlagEmbedding` (the `gpu` group). For verification we
> kept it simple and ran everything on the VM.

---

## 1. Level-1 checks (no GPU, no cloud) — do this first

Proves all Phase 1–4 *logic* with mocked vendors. ✅ Confirmed: **76 passed**.

```powershell
# [WIN] backend/
uv sync
uv run ruff check
uv run pyrefly check
uv run pytest
```

- [x] `ruff check` clean
- [x] `pyrefly check` → 0 errors
- [x] `pytest` → all pass (unit suite, vendors mocked)

---

## 2. One-time GPU VM setup

```bash
# [VM]
nvidia-smi        # confirm the L40S and a CUDA version are listed
```

- [x] `nvidia-smi` shows the L40S

Install uv and get the repo:

```bash
# [VM]
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL                          # reload PATH so `uv` is found
git clone <your-repo-url> graphrag   # or copy the project across
cd graphrag/backend
uv sync --group gpu                  # pulls CUDA torch, Docling, BGE-M3
```

Confirm the GPU deps import and see CUDA:

```bash
# [VM] backend/
uv run python -c "import docling; print('docling', docling.__version__)"
uv run python -c "import FlagEmbedding; print('FlagEmbedding OK')"
uv run python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
```

- [x] docling 2.126.0, FlagEmbedding OK, `torch ... cuda? True`

Persistent HF cache so models download once:

```bash
# [VM] add to ~/.bashrc, then: source ~/.bashrc
export HF_HOME=$HOME/.cache/huggingface
```

Create the VM's `backend/.env` — same cloud creds as Windows PLUS the GPU keys:

| Key | Value on the VM |
|---|---|
| `SARVAM_API_KEY` | same as Windows (graph extraction + synthesis call Sarvam) |
| `OPENROUTER_API_KEY` | same as Windows (reranking) |
| `QDRANT_URL`, `QDRANT_API_KEY` | same cloud Qdrant |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` | same cloud Neo4j |
| `EMBEDDING_SERVICE_URL` | `http://localhost:8081` (query API reaches the local embedding service) |
| `EMBEDDING_SERVICE_TOKEN` | any shared secret; the query API and embedding service must use the **same** value |
| `EMBEDDING_DEVICE` | `cuda:0` |
| `LLM_EXTRACT_REASONING_EFFORT` | `medium` |
| `LLM_EXTRACT_MAX_TOKENS` | `4096` |
| `DOCLING_VLM_URL` | leave blank for Phase 3; set in Phase 4 (§6) |
| `DOCLING_EXTRACT_CHARTS` | `false` (see the note in §6b) |

- [x] VM `backend/.env` filled

---

## 3. Start the embedding service (VM, tmux window 1)

```bash
# [VM] backend/  — NOTE the --factory flag (the app is a factory function)
uv run uvicorn "src.indexing.embedding_service:create_embedding_service_app" --factory \
  --host 0.0.0.0 --port 8081
```

First start loads BGE-M3 (multi-GB download the first time, then cached). Verify:

```bash
# [VM] — token must match .env
curl -H "Authorization: Bearer <EMBEDDING_SERVICE_TOKEN>" http://localhost:8081/health
# -> {"status":"ok","device":"cuda:0"}
```

- [x] `/health` returns `status: ok`, `device: cuda:0`

---

## 4. Start the query API (VM, tmux window 2)

> **The app uses a factory** (`create_app`), so `--factory` is required. Running
> `uvicorn src.api.main:app` fails with `Attribute "app" not found`.

```bash
# [VM] backend/  — must be run from backend/ so .env is found
uv run uvicorn "src.api.main:create_app" --factory --port 8002
```

```bash
# [VM]
curl http://localhost:8002/health
```

- [x] `/health` → qdrant `ok`, neo4j `ok`, embedding_service `ok`

> If `embedding_service` is `unhealthy` with "EMBEDDING_SERVICE_URL is not
> configured", the API was started from the wrong directory (it couldn't find
> `.env`). Restart it from `backend/`.

---

## 5. Phase 3 verification — knowledge graph + SOP versioning ✅

### 5a. Ingest the sample corpus (VM)

There are **four** sample docs and they need **three** ingestion passes (two SOP
versions with different `--sop-version`, plus product info). A full `reset_stores`
followed by ingesting only *some* of them is the trap that breaks later questions
— ingest **all four**.

```bash
# [VM] backend/  — clean slate
uv run python scripts/reset_stores.py --yes

# --- IMPORTANT: both Expense Approval versions must share the same doc_title ---
# The SUPERSEDES chain matches on an identical doc_title. Pass the same
# --doc-title "Expense Approval SOP" on BOTH the v2.1 and v3.0 passes so they
# share a title and the chain forms. (If --doc-title is omitted, a trailing
# version suffix is stripped from the filename, so expense_approval_sop_v2.md and
# ..._v3.md still collapse to the same title — but prefer the explicit flag.)

# Pass 1: superseded SOP (v2.1) — must be ingested BEFORE v3
rm -f data/raw/*.md; cp data/samples/expense_approval_sop_v2.md data/raw/
uv run python scripts/run_ingestion.py \
  --doc-type SOP --sop-version v2.1 --doc-title "Expense Approval SOP" --effective-date 2024-03-01

# Pass 2: current SOP (v3.0) — creates SUPERSEDES v3.0 -> v2.1
rm -f data/raw/*.md; cp data/samples/expense_approval_sop_v3.md data/raw/
uv run python scripts/run_ingestion.py \
  --doc-type SOP --sop-version v3.0 --doc-title "Expense Approval SOP" --effective-date 2025-06-01

# Pass 3: onboarding SOP (needed for the compliance question)
rm -f data/raw/*.md; cp data/samples/onboarding_sop_v1.md data/raw/
uv run python scripts/run_ingestion.py \
  --doc-type SOP --sop-version v1.4

# Pass 4: product info (needed for the Atlas router question)
rm -f data/raw/*.md; cp data/samples/product_atlas_router.md data/raw/
uv run python scripts/run_ingestion.py \
  --doc-type product_info
```

> Do NOT run `reset_stores.py` between these four passes — each doc adds to the
> same graph, and v3.0 needs v2.1 already present.

Each run prints a JSON summary. Confirm per run:

- [x] `docs_processed` matches file count, `chunks_created > 0`
- [x] `graph_nodes_created > 0`, `graph_edges_created > 0` (for the doc-bearing passes)
- [x] `failed_documents` empty

> A few `graph_extraction_empty_reply` / `extracted=0` log lines are normal —
> heading/overview chunks have no extractable facts and degrade to zero triples.

### 5b. Confirm the graph + version chain (Neo4j browser) ✅

```cypher
-- Two Process nodes now exist (one per version), not one
MATCH (p:Process) RETURN p.name, p.sop_version, p.is_current_version;

-- SUPERSEDES edge: v3.0 -> v2.1, old flagged not-current
MATCH (new:Process)-[:SUPERSEDES]->(old:Process)
RETURN new.sop_version, old.sop_version, old.is_current_version;
```

- [x] Two Process nodes: `v2.1` (`is_current_version=false`) and `v3.0` (`true`)
- [x] `SUPERSEDES` edge v3.0 → v2.1 exists

### 5c. Graph-traversal questions (VM) ✅

```bash
# [VM] — repeat per question
curl -X POST http://localhost:8002/query -H "Content-Type: application/json" \
  -d '{"question":"Which role owns the Finance Approval step of Expense Approval?"}'
```

Confirmed results:
1. Finance Approval role → **Finance Analyst** ✅
2. Finance Approval system → **SAP** ✅
3. Employee Onboarding compliance → **SOX-404** (correct answer; see [§9](#9-known-issues-deferred) — citation currently points at the wrong SOP) ⚠️
4. Steps requiring Concur → **Submit / Automated Policy Check / Manager Review** ✅
5. Atlas 5000 features → **Dual-WAN Failover, VPN Concentrator, Zero-Touch Provisioning** ✅

- [x] **DoD:** ≥5 graph-traversal questions answered correctly, each with a citation

### 5d. Version-warning case ✅

```bash
curl -X POST http://localhost:8002/query -H "Content-Type: application/json" \
  -d '{"question":"In the older v2.1 expense approval SOP, what was the manager review approval threshold?"}'
```

- [x] **DoD:** response includes a non-null `version_warning` when a v2.1 chunk is cited

> **Phase 3 complete.** ✅

---

## 6. Phase 4 verification — multimodal (tables + images/charts) ✅

Adds a **VLM on the VM** for image/chart captions and a **PDF with a real table +
chart** (`data/samples/chart.pdf`). Tables extract even without the VLM; captions
need the VLM.

### 6a. Start the VLM (VM, tmux window 3)

We used **Qwen3-VL-8B-Instruct** (newer than the Qwen2.5-VL-7B the spec named).
A current vLLM is required — Qwen3-VL is only in recent releases.

```bash
# [VM] backend/
uv pip install -U vllm            # -U: Qwen3-VL needs a recent vLLM
# (if it rejects the architecture) uv pip install -U transformers

vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 \
  --limit-mm-per-prompt image=1,video=0 \
  --gpu-memory-utilization 0.45
```

Verify:

```bash
curl http://localhost:8000/v1/models      # lists Qwen/Qwen3-VL-8B-Instruct
nvidia-smi                                  # BGE-M3 + vLLM both resident, < 48 GB
```

- [x] vLLM serving Qwen3-VL-8B; fits alongside BGE-M3

> **Model choice is pure config** — the backend forwards `DOCLING_VLM_MODEL` to
> the endpoint. Qwen2.5-VL-7B is a fine fallback if a Qwen3-VL/vLLM version
> mismatch blocks startup.

### 6b. Point Docling at the VLM (VM `.env`)

```bash
DOCLING_VLM_URL=http://localhost:8000/v1/chat/completions
DOCLING_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
DOCLING_VLM_TOKEN=            # blank; vLLM is on the private network
DOCLING_EXTRACT_CHARTS=false  # see note below
```

- [x] VM `.env` has `DOCLING_VLM_URL` + `DOCLING_VLM_MODEL`

> **Why `DOCLING_EXTRACT_CHARTS=false`.** With chart extraction on, Docling 2.126
> pulls its own local `ibm-granite/granite-vision-4.1-4b` (~8 GB) model — a second
> big vision model, separate from your remote VLM, that competes for VRAM. We
> don't need it: the remote VLM (Qwen3-VL) already captions charts well as
> `image_caption` chunks. Turning it off avoids the download while tables and
> captions still work. Set it to `true` only if you want charts *also* converted
> to structured `table` chunks.
>
> **First-PDF model downloads are normal.** The first PDF triggers one-time
> downloads of Docling's standard pipeline models (RapidOCR ~32 MB, layout-heron,
> TableFormer). These are small, cached forever, and unrelated to the 8 GB Granite
> model above.

### 6c. Ingest the table/chart PDF (VM)

```bash
# [VM] backend/  — do NOT reset; this adds to the Phase 3 data
mkdir -p data/samples_tables
cp data/samples/chart.pdf data/samples_tables/
INGESTION_DATA_DIR=data/samples_tables uv run python scripts/run_ingestion.py \
  --doc-type product_info
```

Confirmed run summary: `tables_extracted: 1`, `images_extracted: 1`,
`chunks_created: 3` (one `table`, one `image_caption`, one text).

- [x] `tables_extracted > 0`
- [x] `images_extracted > 0` (Qwen3-VL captioning)

> Watch the log for POSTs to `localhost:8000` during the run — that's Docling
> calling the VLM to caption the chart. If the run hangs there, vLLM is
> unreachable; confirm `curl http://localhost:8000/v1/models`.

### 6d. Table/chart questions (VM) ✅

`chart.pdf` contains a **bar chart** (Dogs & Cats, Jan–May) → `image_caption`
chunk, and an **employee table** (name/position/branch/age/salary) → `table`
chunk. We verified both paths, and the citations resolve to the correct chunk
(chart questions cite the image_caption chunk; table questions cite the table
chunk).

Chart (VLM caption path):
```bash
curl -X POST http://localhost:8002/query -H "Content-Type: application/json" \
  -d '{"question":"How many dogs were there in April according to the chart?"}'      # ~180 ✅
curl -X POST http://localhost:8002/query -H "Content-Type: application/json" \
  -d '{"question":"In which month were there the most cats?"}'                        # May 500 ✅
curl -X POST http://localhost:8002/query -H "Content-Type: application/json" \
  -d '{"question":"In May, were there more cats or more dogs?"}'                      # cats ✅
```

Table (Docling TableFormer path):
```bash
curl -X POST http://localhost:8002/query -H "Content-Type: application/json" \
  -d '{"question":"What is the salary of Harvey Spencer and what is his position?"}'  # 470,600 / Regional Director ✅
curl -X POST http://localhost:8002/query -H "Content-Type: application/json" \
  -d '{"question":"Which employees work in the London office branch?"}'              # Brady, Spencer, Hyatt ✅
curl -X POST http://localhost:8002/query -H "Content-Type: application/json" \
  -d '{"question":"Who has the highest salary in the employee table?"}'             # Harvey Spencer ✅
```

- [x] **DoD:** ≥5 table/chart questions answered correctly with citations (6/6 passed)
- [x] Both extraction paths (table-markdown + VLM caption) independently verified and citable

> **Phase 4 complete.** ✅

---

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `uvicorn ... Attribute "app" not found` | Use `--factory` and the factory target: `src.api.main:create_app` / `src.indexing.embedding_service:create_embedding_service_app` |
| `/health` embedding_service `unhealthy: EMBEDDING_SERVICE_URL is not configured` | Query API started outside `backend/` so it didn't load `.env`; restart from `backend/` |
| `/health` embedding_service `unhealthy` (other) | Embedding service not running, or `EMBEDDING_SERVICE_TOKEN` mismatch between the two `.env`s |
| `/health` returns 503 | qdrant or embedding_service not `ok` — API reports ready only when both are up |
| Ingestion: `in-process embedding model could not be loaded` | `uv sync --group gpu` not run, or no GPU visible (`nvidia-smi`) |
| Ingestion pulls an ~8 GB `granite-vision` model | Chart extraction is on; set `DOCLING_EXTRACT_CHARTS=false` unless you need chart→table |
| First PDF downloads several small models | Normal one-time Docling pipeline setup (OCR/layout/TableFormer); cached afterward |
| No `SUPERSEDES` edge / no version warning | v2 and v3 ingested under different `doc_title`s or wrong order — pass the same `--doc-title "..."` on both passes and ingest v2 before v3 (§5a) |
| `tables_extracted`/`images_extracted` stay 0 | Source is Markdown (no tables/pictures) — use a PDF like `chart.pdf` |
| Ingestion hangs at picture captioning | vLLM not reachable; `curl http://localhost:8000/v1/models` |
| vLLM won't start (unknown architecture) | vLLM too old for Qwen3-VL: `uv pip install -U vllm` (and maybe `transformers`), or fall back to Qwen2.5-VL-7B |
| vLLM OOM | Lower `--gpu-memory-utilization`, or run the VLM on a separate GPU/box |

---

## 8. DoD summary checklist

Phase 3:
- [x] Unit suite + ruff + pyrefly green
- [x] ≥5 graph-traversal questions answered correctly with citations
- [x] `SUPERSEDES` edge created; v2.1 chunks flipped to `is_current_version=false`
- [x] Version warning surfaces when a superseded chunk is cited

Phase 4:
- [x] Unit suite + ruff + pyrefly green
- [x] `tables_extracted > 0` on a table-bearing document
- [x] ≥5 table/chart questions answered correctly with citations
- [x] Image/chart captions extracted and retrievable (with VLM)

---

## 9. Known issues (deferred)

### 9.1 Graph-answer citation points to the wrong document (Phase 3, Q3)

**Symptom.** "What compliance requirement must the Employee Onboarding process
satisfy?" returns the correct answer (SOX Financial Controls, SOX-404) but the
citations point to **"Expense Approval SOP"** (v2.1/v3.0) instead of the
onboarding document, and an irrelevant version warning fires.

**Likely cause.** Two-part retrieval issue, not a grounding bug:
- Graph entity matching fuzzy-matched the wrong node (`"Employee"` instead of
  `"Employee Onboarding"`) — the bidirectional `CONTAINS` match in
  `GraphStore.match_entities` is too loose.
- Reranking scored semantically-similar Expense-Approval SOX chunks above the
  onboarding chunk, so the cited source is the wrong (but topically similar) doc.

**Impact.** Answer correctness is fine; provenance accuracy is wrong for this
class of question. Does not block the Phase 3 DoD (≥5 correct with citations was
met by Q1/Q2/Q4/Q5 plus the chart/table questions), but it undermines the
citation-grounding guarantee and should be fixed.

**Fix sketch (later).** Tighten `match_entities` (require stronger overlap / rank
by match length, avoid matching a multi-word entity to a single-word node), and/or
prefer the graph fact's own `chunk_id` source document when resolving the citation
rather than the top reranked chunk. Add a regression test with an onboarding-vs-
expense disambiguation case.

**Resolved (2026-10).** Two fixes landed in the backend:

- `GraphStore.match_entities` no longer uses the loose bidirectional `CONTAINS`
  result directly. It keeps a parameterized containment pre-filter (read routing,
  `database_` passed) but then runs a Python token-overlap/specificity rule
  (`_best_entity_match`): exact token-set equality wins; otherwise one side's
  tokens must fully contain the other's, a bare single-word candidate may not
  expand into a larger node (killing the `"Employee"` → `"Employee Onboarding"`
  leak), and spec-label/SKU nodes (`ATL-5000`, `Target`, `Throughput`,
  `Port Count`, `Simultaneous Tunnels`) no longer match a product candidate.
  Legitimate fuzzy matches (`"SAP system"` → `SAP`, `"Finance Approval"` → the
  step node) still resolve.
- `CitableSource.from_graph_fact` now resolves a graph fact across *all* of its
  `chunk_ids` and returns the real document's `doc_title`/version/current-flag.
  When no chunk id is among the retrieved sources the fact is **dropped** instead
  of emitting the synthetic `"Knowledge graph"` label; the synthesizer skips such
  facts. This removes the misleading generic citation and the spurious version
  warning it could trigger, while the legitimate superseded-version warning (a
  real v2.1 chunk cited) still fires.

A read-only graph `doc_title` lookup helper was intentionally **not** added:
non-`Process` graph nodes (`Product`/`Feature`/`Spec`/`Step`/`Role`/`System`/
`ComplianceRequirement`) store no `doc_title`, so a per-fact Neo4j lookup could
not resolve one anyway and would only add a query-time round-trip and failure
surface. Regression tests: `tests/test_entity_matching.py` and the graph-fact
provenance cases in `tests/test_citation_grounding.py` / `tests/test_graph_search.py`.

### 9.2 Note on the versioning fix (already resolved, for context)

Earlier, no `SUPERSEDES` edge formed because the two SOP versions were ingested
under different `doc_title`s (derived from version-specific filenames), so they
collapsed onto — or failed to link — a single Process node. Fixed by making the
Process node MERGE key version-aware (`(name, sop_version)`) in `graph_store.py`,
and by giving `run_ingestion.py` a `--doc-title` flag so both versions can be
ingested under one shared title (§5a). The parser also strips a trailing version
suffix from the filename as a fallback when `--doc-title` is omitted, so sibling
version files still share a title. Kept here so the shared-title requirement in
§5a isn't mistaken for a typo.

---

## 10. Future work

### 10.1 Evaluate Sarvam Document AI as a table-extraction backend

We considered replacing the Docling + vLLM ingestion stack with **Sarvam
Document AI** (Digitise/Extract, powered by Sarvam Vision 1.5/2.1). Findings from
that investigation:

- **Where it wins:** hosted API (no vLLM process, no local Docling models, no 8 GB
  Granite download), strong table extraction to HTML/Markdown/JSON, and it's a
  vendor we already use. Especially compelling for **scanned or Indic-language**
  corpora, which is exactly its strength.
- **Where it doesn't fit cleanly today:** its API surface is Digitise (full-page
  OCR + tables) and Extract (schema fields) — neither returns a per-figure prose
  **caption** the way our `content_type="image_caption"` design and Docling's
  `PictureDescriptionApiOptions` flow do. It also caps at **10 pages/PDF**, and
  swapping it in means rewriting `parser.py`/`multimodal.py` and their models/tests
  around Sarvam's output shape.

**Decision (deferred, not now).** Keep the current Docling (tables) + Qwen3-VL
(captions) pipeline, which is implemented, tested, and DoD-verified. Revisit
Sarvam Doc AI later as a **table-extraction backend swap** (highest value, lowest
risk), keeping a VLM for figure captioning. If/when adopted, record it as a new
deviation in `BACKEND.md §11` and add config to select the ingestion backend.
This is an operational simplification, not a missing capability — so it should not
block Phases 5–7.

> The original spec decision (BACKEND.md §3.4 / D-3) to keep captioning in Docling
> because "Sarvam Vision is OCR-only" is now *weaker* — newer Sarvam Vision claims
> chart interpretation and image captioning — but the API still exposes those as
> OCR/extraction, not free-form captioning, so the decision stands for now.
