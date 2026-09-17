# RUNNING.md — How to run and test the backend

> Practical runbook for operating the system across the two-machine topology.
> For *what* the system is, see `SPEC.md`; for the build plan, see `BACKEND.md`.
> This file is the step-by-step "how do I actually run it" guide.

---

## 1. Topology recap

Two machines plus cloud services:

```
Windows box (CPU)                 GPU VM (Ubuntu + L40S)              Cloud
─────────────────                 ──────────────────────             ─────
Query API (FastAPI)  ──HTTP──▶    embedding_service :8081            Qdrant Cloud
  /query /health /ingest          run_ingestion.py (batch, GPU)      Neo4j AuraDB
  router, retrieval,              (Docling + BGE-M3 in-process)      Sarvam / OpenRouter / Exa
  rerank, synthesis, graph
```

- **Ingestion runs ONLY on the GPU VM** (needs GPU for BGE-M3 + Docling). It
  writes chunks to Qdrant Cloud and the graph to Neo4j AuraDB.
- **The query API runs on your box** (CPU-only). It reads the same cloud stores
  and calls the VM's embedding service to embed each incoming question.
- The query API must **never** import `torch`/`FlagEmbedding`/`docling` — those
  live in the `gpu` dependency group, installed only on the VM.

---

## 2. Environment (.env)

`.env` lives in `backend/` and is gitignored. Both machines need the same cloud
credentials. Copy `backend/.env.example` and fill it in.

### Required on both machines
| Key | Notes |
|---|---|
| `SARVAM_API_KEY` | LLM (synthesis, extraction, entity extraction) |
| `QDRANT_URL`, `QDRANT_API_KEY` | Qdrant Cloud |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Neo4j AuraDB |
| `NEO4J_DATABASE` | **AuraDB uses the instance id, not `neo4j`.** Find it with `SHOW DATABASES` (see §6). Wrong value → `DatabaseNotFound`. |
| `OPENROUTER_API_KEY` | Reranking. If absent/placeholder, rerank degrades gracefully to RRF order. |

### Query-API box only
| Key | Notes |
|---|---|
| `EMBEDDING_SERVICE_URL` | URL of the VM embedding service (e.g. `http://<VM_IP>:8081`, or `http://localhost:8081` via SSH tunnel). |
| `EMBEDDING_SERVICE_TOKEN` | Must match the VM's token. |

### GPU VM only
| Key | Notes |
|---|---|
| `EMBEDDING_SERVICE_TOKEN` | Same value as the box. |
| `EMBEDDING_DEVICE` | `cuda:0` on the VM. |

### Extraction tuning (defaults are good)
| Key | Default | Notes |
|---|---|---|
| `LLM_EXTRACT_REASONING_EFFORT` | `medium` | `low`/`null` cause empty extraction replies on content-rich chunks and a sparse graph. Keep `medium`. |
| `LLM_EXTRACT_MAX_TOKENS` | `4096` | Room for reasoning tokens so the JSON isn't truncated. |

> The defaults for the two extraction keys are baked into `config.py`, so you do
> not need to set them unless you want to override. Everything else optional
> uses its documented default when left blank.

---

## 3. One-time setup

### GPU VM
```bash
nvidia-smi                                    # confirm L40S + CUDA
curl -LsSf https://astral.sh/uv/install.sh | sh && exec $SHELL
git clone <repo> graph && cd graph/backend
uv sync --group gpu                           # CUDA torch + FlagEmbedding + docling
export HF_HOME=/opt/hf_cache                  # persist the BGE-M3 model cache
# create .env (see §2)
```

### Windows box
```powershell
cd s:\1-Project\Tushan\final\graphrag\backend
uv sync                                        # CPU deps only
# create .env (see §2)
```

---

## 4. Running the embedding service (GPU VM)

Both apps are FastAPI **factories**, so uvicorn needs `--factory`.

```bash
# quick demo (run inside tmux/screen):
uv run uvicorn src.indexing.embedding_service:create_embedding_service_app \
  --factory --host 0.0.0.0 --port 8081
```

Durable (recommended): run under systemd. Example unit:
```ini
# /etc/systemd/system/embedding.service
[Unit]
Description=BGE-M3 Embedding Service
After=network.target
[Service]
WorkingDirectory=/home/<user>/graph/backend
Environment=HF_HOME=/opt/hf_cache
ExecStart=/home/<user>/.local/bin/uv run uvicorn src.indexing.embedding_service:create_embedding_service_app --factory --host 0.0.0.0 --port 8081
Restart=on-failure
[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload && sudo systemctl enable --now embedding
curl -H "Authorization: Bearer <TOKEN>" http://localhost:8081/health
# -> {"status":"ok","device":"cuda:0"}
```

> Security: keep 8081 on a private network / firewalled to the box, or reach it
> via SSH tunnel: `ssh -N -L 8081:localhost:8081 <user>@<VM_IP>`.

---

## 5. Ingestion (GPU VM)

Ingestion reads everything in `data/raw/`. The `--sop-version` flag stamps every
file in that folder for the run, so ingest versioned SOPs one file at a time.
The committed sample corpus lives in `data/samples/`.

### Clean slate (optional but recommended for a fresh test)
```bash
uv run python scripts/reset_stores.py --yes          # wipes Qdrant collection + Neo4j graph
# or one store only:
uv run python scripts/reset_stores.py --yes --only qdrant
uv run python scripts/reset_stores.py --yes --only neo4j
```

### Ingest the sample corpus (ORDER MATTERS for the superseded case)
Ingest Expense Approval **v2.1 before v3.0** so v3.0 can create the
`SUPERSEDES` edge to v2.1 and flip v2.1 to not-current.

```bash
cd ~/graph/backend

# 1. Expense Approval v2.1 (older version)
rm -f data/raw/*.md; cp data/samples/expense_approval_sop_v2.md data/raw/
uv run python scripts/run_ingestion.py --doc-type SOP --sop-version v2.1 --effective-date 2024-03-01

# 2. Expense Approval v3.0 (current; supersedes v2.1)
rm -f data/raw/*.md; cp data/samples/expense_approval_sop_v3.md data/raw/
uv run python scripts/run_ingestion.py --doc-type SOP --sop-version v3.0 --effective-date 2025-06-01

# 3. Onboarding SOP
rm -f data/raw/*.md; cp data/samples/onboarding_sop_v1.md data/raw/
uv run python scripts/run_ingestion.py --doc-type SOP --sop-version v1.4 --effective-date 2024-09-15

# 4. Product info (no version flag)
rm -f data/raw/*.md; cp data/samples/product_atlas_router.md data/raw/
uv run python scripts/run_ingestion.py --doc-type product_info
```

> Do NOT run `reset_stores.py` between these four — each doc adds to the same
> graph, and v3.0 needs v2.1 already present.

### What a healthy ingestion looks like
Each run prints an `IngestionSummary`. For a single SOP expect roughly:
- `chunks_created` > 0, `failed_documents: []`
- `graph_nodes_created` ~10-14, `graph_edges_created` ~10-14
- Per-chunk logs `graph_triples_extracted ... extracted=N` with N mostly 2-4;
  only truly factless chunks (an "Overview" heading) should be empty.

If many content chunks log `graph_extraction_empty_reply`, extraction reasoning
is too low — confirm `LLM_EXTRACT_REASONING_EFFORT=medium`.

---

## 6. Verifying the graph (Neo4j Browser / Aura console)

**Find your database name** (needed for `NEO4J_DATABASE`):
```cypher
SHOW DATABASES;
```
Use the `standard` / `online` database name (on AuraDB this is the instance id).

**All edges** (sanity of shapes):
```cypher
MATCH (a)-[r]->(b)
RETURN labels(a)[0] AS from, a.name AS subject, type(r) AS rel,
       labels(b)[0] AS to, b.name AS object
ORDER BY rel;
```
Expect a mix of `HAS_STEP` (Process→Step), `OWNED_BY` (Step→Role),
`REQUIRES` (Step→System), `MUST_SATISFY` (Process→ComplianceRequirement).

**Process nodes are unified** (one per document, not fragmented):
```cypher
MATCH (p:Process)
RETURN p.name, p.doc_title, p.sop_version, p.is_current_version, size(p.chunk_ids);
```
Expect one Process per document/version. After ingesting v3.0, the v2.1 row
should show `is_current_version = false`.

**Version chain**:
```cypher
MATCH (new:Process)-[:SUPERSEDES]->(old:Process)
RETURN new.sop_version AS current, old.sop_version AS superseded, old.is_current_version AS old_flag;
```
Expect: `current="v3.0"`, `superseded="v2.1"`, `old_flag=false`.

---

## 7. Running the query API (Windows box)

Make sure the VM embedding service is reachable (direct IP or SSH tunnel) and the
graph is populated. Then:

```powershell
cd s:\1-Project\Tushan\final\graphrag\backend
uv run uvicorn src.api.main:create_app --factory --host 127.0.0.1 --port 8000
```

(Run this in your own terminal; it is a long-running server.)

---

## 8. Testing the API

### Health — verifies Qdrant + Neo4j + embedding service
```powershell
curl http://127.0.0.1:8000/health
```
Expect `status: "ok"` with all three components `ok`. If `neo4j` is `unhealthy`,
check `NEO4J_DATABASE`. If `embedding_service` is `unhealthy`, the VM/tunnel is
unreachable.

### Graph-traversal questions (answerable only via the graph)
```powershell
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" `
  -d '{\"question\": \"Which role owns the Finance Approval step and which system does it require?\", \"conversation_id\": null}'
```
Other good ones:
- "What compliance requirement must the Employee Onboarding process satisfy?"
- "Which steps require the Concur system?"
- "What features and specs does the Atlas 5000 Router have?"

### Superseded-SOP version warning
```powershell
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" `
  -d '{\"question\": \"What was the manager review threshold in the v2.1 expense approval SOP?\", \"conversation_id\": null}'
```
If the answer cites a v2.1 chunk, `version_warning` should be non-null.

### Ingestion status (last VM run summary)
```powershell
curl http://127.0.0.1:8000/ingest
```

### What to check in every `/query` response
- `answer` with inline `[n]` markers
- `citations[]` — each is chunk-based (`reference.chunk_id`, `doc_title`)
- `version_warning` — set when a superseded SOP chunk is cited
- `tool_trace` — `["internal_search"]` (the tool-calling router arrives in Phase 5)

---

## 9. Unit tests (offline, no live services, no VM)

```powershell
cd backend
uv run pytest              # all vendors mocked; excludes integration marker
uv run ruff check
uv run pyrefly check
```

---

## 10. Known behavior with the current setup

- **OpenRouter placeholder key** → reranking fails and falls back to RRF order
  (logged as `rerank_fallback_to_rrf`). Answers still work; add a real key to
  exercise reranking.
- **Exa** is not used in the query path until Phase 5; its placeholder is fine.
- **AuraDB free tier auto-pauses when idle** — if `/health` shows neo4j
  `unhealthy` after a break, resume the instance in the Aura console.
- **Extraction latency** — each chunk is one Sarvam call (~10-20s at `medium`).
  Ingestion is a batch job, so this is expected; it does not affect query latency.

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run scripts from `backend/`. Scripts also self-bootstrap `sys.path`; `git pull` if you see this. |
| `Neo.ClientError.Database.DatabaseNotFound` | `NEO4J_DATABASE` wrong. Run `SHOW DATABASES`, use the instance id. |
| `document_ingestion_failed ... Sarvam returned an empty answer` | Fixed: extractor now tolerates empty replies per chunk. `git pull` if seen. |
| Sparse graph (few edges), many `graph_extraction_empty_reply` | `LLM_EXTRACT_REASONING_EFFORT` too low; use `medium`. |
| Process split into multiple nodes | Fixed: Process name is canonicalized to `doc_title`. Re-ingest after `git pull`. |
| `/health` embedding_service `unhealthy` | VM service down or `EMBEDDING_SERVICE_URL`/token wrong; check tunnel. |
