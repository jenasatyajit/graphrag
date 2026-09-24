# VERIFICATION.md — Verifying Phases 1–4 end to end

> A tickable, copy-paste guide to bring up the GPU VM, connect this backend to
> it, ingest the sample corpus, and confirm each phase's Definition of Done
> (DoD). Follow it top to bottom. Phase 3 first (no VLM needed), then Phase 4
> (adds the VLM + a table/chart document).

Legend: run commands from the shown directory. `[WIN]` = your Windows box,
`[VM]` = the remote Ubuntu 22.04 GPU box (SSH). "cloud" services (Qdrant, Neo4j,
Sarvam, OpenRouter) are already configured in `backend/.env`.

---

## 0. Topology recap (why the steps look like this)

```
   [WIN] Windows box                          [VM] Ubuntu 22.04 + L40S 48GB
┌──────────────────────────┐   SSH tunnel   ┌──────────────────────────────────┐
│ FastAPI query API         │──:8081────────▶│ embedding_service (BGE-M3)        │
│  uv sync (CPU only)        │                │   GET /health  POST /embed        │
│  reads backend/.env        │                │ run_ingestion.py (Docling+BGE-M3) │
│  EMBEDDING_SERVICE_URL     │                │   writes chunks+graph to cloud    │
│    = http://localhost:8081 │                │ [Phase 4] vLLM Qwen2.5-VL :8000   │
└───────────┬────────────────┘                └──────────────┬───────────────────┘
            │ cloud                                           │ cloud
            ▼                                                 ▼
      Qdrant Cloud  ◀───────────────────────────────────  Neo4j AuraDB
```

- **Ingestion runs on the VM** (needs Docling + BGE-M3 in-process). It writes to
  cloud Qdrant + Neo4j.
- **The query API runs on Windows** and only needs the embedding service reachable
  over an SSH tunnel. It reads the same cloud stores the VM wrote to.
- The API box must **never** install `torch`/`docling`/`FlagEmbedding` — those live
  in the `gpu` dependency group, installed only on the VM.

---

## 1. Level-1 checks (no GPU, no cloud) — do this first

These prove all Phase 1–4 *logic* with mocked vendors.

```powershell
# [WIN] backend/
uv sync
uv run ruff check
uv run pyrefly check
uv run pytest
```

- [ ] `ruff check` clean
- [ ] `pyrefly check` → 0 errors
- [ ] `pytest` → all pass (unit suite, vendors mocked)

If green, the automated portion of **every** phase DoD is satisfied. The rest of
this doc covers the manual "demoable" DoD items.

---

## 2. One-time GPU VM setup

SSH in, then:

```bash
# [VM]
nvidia-smi        # confirm the L40S and a CUDA version are listed
```

- [ ] `nvidia-smi` shows the L40S

Install uv and clone the repo (skip clone if already present):

```bash
# [VM]
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL                          # reload PATH so `uv` is found
git clone <your-repo-url> graphrag   # or copy the project across
cd graphrag/backend
```

Install the GPU dependency group (this pulls a CUDA `torch`, Docling, BGE-M3):

```bash
# [VM] backend/
uv sync --group gpu
```

- [ ] `uv sync --group gpu` completed (downloads a multi-GB CUDA torch wheel)

Set a persistent Hugging Face cache so BGE-M3 (~2 GB) isn't re-downloaded:

```bash
# [VM] add to ~/.bashrc, then `source ~/.bashrc`
export HF_HOME=$HOME/.cache/huggingface
```

Create the VM's `.env`. It needs the **same cloud creds** as your Windows `.env`
PLUS the GPU-side keys. Copy your Windows `backend/.env` to the VM and edit:

```bash
# [VM] backend/
cp .env.example .env
# then fill it — see the table below
```

| Key | Value on the VM |
|---|---|
| `SARVAM_API_KEY` | same as Windows (graph extraction calls Sarvam) |
| `OPENROUTER_API_KEY` | same as Windows |
| `QDRANT_URL`, `QDRANT_API_KEY` | same cloud Qdrant as Windows |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` | same cloud Neo4j |
| `EMBEDDING_DEVICE` | `cuda:0` |
| `EMBEDDING_SERVICE_TOKEN` | a shared secret — **must match** the Windows `.env` (e.g. `saddle-hen-fabric`) |
| `LLM_EXTRACT_REASONING_EFFORT` | `medium` |
| `LLM_EXTRACT_MAX_TOKENS` | `4096` |
| `DOCLING_VLM_URL` | leave blank for Phase 3; set in Phase 4 (§6) |

- [ ] VM `backend/.env` has cloud creds + `EMBEDDING_DEVICE=cuda:0` + matching `EMBEDDING_SERVICE_TOKEN`

> The VM's `EMBEDDING_SERVICE_URL` is irrelevant (the VM *is* the service). Only
> Windows uses that key.

---

## 3. Start the embedding service on the VM

```bash
# [VM] backend/
uv run uvicorn src.indexing.embedding_service:create_embedding_service_app --factory \
  --host 0.0.0.0 --port 8081
```

First start loads BGE-M3 (multi-GB download the first time; then cached). Wait
until it logs startup complete. Verify locally on the VM:

```bash
# [VM] another shell — token must match .env
curl -H "Authorization: Bearer saddle-hen-fabric" http://localhost:8081/health
# -> {"status":"ok","device":"cuda:0"}
```

- [ ] `/health` on the VM returns `status: ok`, `device: cuda:0`

> For a long-lived setup, run this under systemd or tmux so it survives your SSH
> session. For a demo, a dedicated tmux window is fine. Do **not** expose 8081 to
> the public internet — reach it via the SSH tunnel in §4.

---

## 4. Open the SSH tunnel from Windows

This maps `localhost:8081` on Windows to `localhost:8081` on the VM, matching your
`EMBEDDING_SERVICE_URL`. Keep this shell open.

```powershell
# [WIN]
ssh -N -L 8081:localhost:8081 <user>@<vm-host>
```

Verify from Windows (new shell):

```powershell
# [WIN]
curl -H "Authorization: Bearer saddle-hen-fabric" http://localhost:8081/health
```

- [ ] Windows can reach the embedding service through the tunnel (`status: ok`)

---

## 5. Phase 3 verification — knowledge graph + SOP versioning

### 5a. Ingest the sample corpus (on the VM)

The samples live in `backend/data/samples/`. Ingest **v2 first, then v3** so the
`SUPERSEDES` edge and `is_current_version=false` flip happen. The order matters.

```bash
# [VM] backend/  — clean slate first (optional but recommended)
uv run python scripts/reset_stores.py --yes

# Ingest the older SOP version (v2.1)
INGESTION_DATA_DIR=data/samples_v2 uv run python scripts/run_ingestion.py \
  --doc-type SOP --sop-version v2.1
```

> The ingestion CLI ingests **every** file in `INGESTION_DATA_DIR` with one
> `--sop-version`. Since v2 and v3 need different versions, ingest them in two
> passes from two folders. Create them once on the VM:
>
> ```bash
> # [VM] backend/
> mkdir -p data/samples_v2 data/samples_v3
> cp data/samples/expense_approval_sop_v2.md data/samples_v2/
> cp data/samples/expense_approval_sop_v3.md data/samples_v3/
> cp data/samples/onboarding_sop_v1.md       data/samples_v3/   # current SOP
> cp data/samples/product_atlas_router.md     data/samples_v3/   # product info
> ```
>
> Then ingest product info separately with its own doc-type:
>
> ```bash
> # v2.1 superseded SOP
> INGESTION_DATA_DIR=data/samples_v2 uv run python scripts/run_ingestion.py \
>   --doc-type SOP --sop-version v2.1
> # v3.0 current SOP + onboarding (both SOPs)
> rm data/samples_v3/product_atlas_router.md
> INGESTION_DATA_DIR=data/samples_v3 uv run python scripts/run_ingestion.py \
>   --doc-type SOP --sop-version v3.0
> # product info (no version)
> mkdir -p data/samples_prod && cp data/samples/product_atlas_router.md data/samples_prod/
> INGESTION_DATA_DIR=data/samples_prod uv run python scripts/run_ingestion.py \
>   --doc-type product_info
> ```

Each run prints a JSON summary. Confirm:

- [ ] `docs_processed` matches the file count, `chunks_created > 0`
- [ ] `graph_nodes_created > 0` and `graph_edges_created > 0`
- [ ] `failed_documents` is empty

### 5b. Confirm the graph + version chain in Neo4j

In the Neo4j AuraDB browser (or `cypher-shell`), run:

```cypher
// SUPERSEDES edge exists (v3.0 -> v2.1)
MATCH (new:Process)-[:SUPERSEDES]->(old:Process) RETURN new, old;

// Expense Approval steps, owners, and required systems
MATCH (p:Process {name:"Expense Approval SOP"})-[:HAS_STEP]->(s:Step)
OPTIONAL MATCH (s)-[:OWNED_BY]->(r:Role)
OPTIONAL MATCH (s)-[:REQUIRES]->(sys:System)
RETURN s.name, r.name, sys.name;
```

- [ ] A `SUPERSEDES` edge from v3.0 to v2.1 exists
- [ ] Steps map to roles/systems (e.g. Finance Approval → Finance Analyst → SAP)

### 5c. Start the query API on Windows and ask graph questions

```powershell
# [WIN] backend/ — tunnel (§4) must be open
uv run uvicorn src.api.main:app --port 8000
```

```powershell
# [WIN] another shell
curl http://localhost:8000/health
```

- [ ] `/health` → qdrant `ok`, neo4j `ok`, embedding_service `ok`

Ask the five graph-traversal questions (from `data/samples/README.md`). Each
should answer correctly **with a citation**:

```powershell
# [WIN] repeat for each question
$body = '{"question":"Which role owns the Finance Approval step of Expense Approval?"}'
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d $body
```

Questions to run:
1. Which role owns the Finance Approval step of Expense Approval? → Finance Analyst
2. Which system does the Finance Approval step require? → SAP
3. What compliance requirement must the Employee Onboarding process satisfy?
4. Which steps require the Concur system?
5. What features does the Atlas 5000 Router have?

- [ ] **DoD:** ≥5 graph-traversal questions answered correctly, each with a citation

### 5d. Version-warning case

Ask something answered by the **superseded v2.1** content (e.g. the old $500
manager-review threshold rather than v3's $1000):

```powershell
$body = '{"question":"What is the manager review approval threshold in the older expense approval SOP?"}'
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d $body
```

- [ ] **DoD:** response includes a non-null `version_warning` when a v2.1 chunk is cited

> **Phase 3 complete** when 5a–5d are all ticked.

---

## 6. Phase 4 verification — multimodal (tables + images/charts)

Phase 4 adds two things: a **VLM on the VM** for image/chart captions, and a
**document with a real table** (the Markdown samples have none). Tables are
extracted even without the VLM; image/chart captions need the VLM.

### 6a. Start the VLM on the VM (vLLM + Qwen2.5-VL-7B)

Run in a separate tmux window on the VM. Flags re-verified against current vLLM
docs (2026); adjust `--gpu-memory-utilization` so BGE-M3 + ingestion coexist on
the 48 GB card.

```bash
# [VM] backend/
uv pip install "vllm>0.7.2"
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 \
  --limit-mm-per-prompt image=1,video=0 \
  --gpu-memory-utilization 0.45
```

Verify (vLLM accepts `api_key="EMPTY"` by default; protect at the network layer):

```bash
# [VM] another shell
curl http://localhost:8000/v1/models
```

- [ ] vLLM lists `Qwen/Qwen2.5-VL-7B-Instruct`
- [ ] `nvidia-smi` shows both BGE-M3 and vLLM resident, under 48 GB total

### 6b. Point Docling at the VLM (VM `.env`)

Add to the VM's `backend/.env`:

```bash
DOCLING_VLM_URL=http://localhost:8000/v1/chat/completions
DOCLING_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
DOCLING_VLM_TOKEN=            # leave blank; vLLM is behind the private network
```

- [ ] VM `.env` has `DOCLING_VLM_URL` + `DOCLING_VLM_MODEL`

> With `DOCLING_VLM_URL` set, ingestion enables `do_picture_description`,
> `do_chart_extraction`, and image extraction. Blank = tables only, captions
> skipped (graceful degradation).

### 6c. Add a table/chart document

The committed samples are Markdown with no tables, so the table/chart DoD needs at
least one **PDF containing a table** (and ideally a chart). Options:

- Drop a real SOP/product PDF that has a table into `data/raw/` on the VM, **or**
- Create a small test PDF with one table and one bar chart.

Place it in a folder and ingest it:

```bash
# [VM] backend/
mkdir -p data/samples_tables   # put your table PDF here
INGESTION_DATA_DIR=data/samples_tables uv run python scripts/run_ingestion.py \
  --doc-type product_info
```

Confirm the run summary reports multimodal extraction:

- [ ] `tables_extracted > 0` (and `images_extracted > 0` if the PDF has figures)

### 6d. Ask table/chart questions on Windows

Ask ≥5 questions whose answers live **in the table/chart**, not the prose:

```powershell
# [WIN] example
$body = '{"question":"<a question whose answer is a specific table cell value>"}'
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d $body
```

- [ ] **DoD:** ≥5 table/chart questions answered correctly, each with a citation
      whose `content_type` traces to a `table` or `image_caption` chunk

> **Phase 4 complete** when 6a–6d are all ticked.

---

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `/health` embedding_service `unhealthy` | Tunnel (§4) not open, service not running, or token mismatch between the two `.env` files |
| `/health` returns 503 | qdrant or embedding_service not `ok` — the API only reports ready when both are up |
| Ingestion: `The in-process embedding model could not be loaded` | `uv sync --group gpu` not run, or no GPU visible; check `nvidia-smi` |
| Ingestion writes 0 graph edges | Sarvam key/quota issue, or `LLM_EXTRACT_REASONING_EFFORT` set to null (use `medium`) |
| No `version_warning` ever | v2 and v3 were not ingested as separate versions, or ingested in the wrong order (v2 must come before v3) |
| `tables_extracted` stays 0 | Source is Markdown (no tables) — use a PDF with a real table |
| Image captions empty | `DOCLING_VLM_URL` blank, or vLLM not reachable from the VM |
| vLLM OOM | Lower `--gpu-memory-utilization`, or run the VLM on a separate GPU/box |

---

## 8. DoD summary checklist

Phase 3:
- [ ] Unit suite + ruff + pyrefly green
- [ ] ≥5 graph-traversal questions answered correctly with citations
- [ ] `SUPERSEDES` edge created; v2.1 chunks flipped to `is_current_version=false`
- [ ] Version warning surfaces when a superseded chunk is cited

Phase 4:
- [ ] Unit suite + ruff + pyrefly green
- [ ] `tables_extracted > 0` on a table-bearing document
- [ ] ≥5 table/chart questions answered correctly with citations
- [ ] (with VLM) image/chart captions extracted and retrievable
