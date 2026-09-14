# Phase 1 Backend

## Relevant files

```text
backend/
├── scripts/run_ingestion.py
├── src/
│   ├── agent/router.py                 # Phase 1 RAG orchestration seam
│   ├── api/{config,main,routes,schemas}.py
│   ├── generation/{synthesizer,provenance}.py
│   ├── indexing/{embedding_service,embeddings,vector_store}.py
│   └── ingestion/{parser,chunker}.py
└── tests/
    ├── test_api.py
    ├── test_chunker.py
    ├── test_citation_grounding.py
    ├── test_embeddings.py
    ├── test_parser.py
    ├── test_router.py
    └── test_vector_store.py
```

## Core contracts

### Chunk payload

`Chunk` in `src/ingestion/chunker.py` implements the SPEC payload: document/chunk IDs, document title/type, content type, text, page, SOP version/effective date/current flag, entity IDs, and internal provenance. `step_order` retains SOP procedure ordering.

### Embedding contract

```json
{
  "dense": [[0.1, 0.2]],
  "sparse": [{"indices": [12, 98], "values": [0.7, 0.3]}]
}
```

Sparse indices are validated as non-negative, unique, and ascending. `InProcessBGEM3` and `HttpEmbeddingClient` both return this model.

### Query response

`POST /query` returns `answer`, structured `citations`, `used_web_search=false`, `version_warning=null`, and `tool_trace=["internal_search"]`. Conversation memory and routing are intentionally deferred to Phase 5.

## Ingestion behavior

Run it from `backend/` as `uv run python -m scripts.run_ingestion --doc-type <type>`. `run_ingestion.py` requires an explicit `--doc-type` so metadata is never guessed from filenames. It processes supported files independently, batches embeddings, writes both vector types to Qdrant, records failed filenames, and atomically replaces the last-run JSON summary. In the two-machine topology, `INGESTION_SUMMARY_PATH` must refer to a shared mount or the JSON must be copied to the API host after each run.

## Citation enforcement

The synthesizer numbers retrieved chunks before calling Sarvam. `ground_internal_answer` then segments the generated response and keeps only claims containing valid source markers. Invalid, missing, or out-of-range citations are flagged internally and omitted. A pure insufficient-context response may contain no citations because it asserts no retrieved fact.

## API endpoints

| Method | Path | Phase 1 behavior |
|---|---|---|
| `GET` | `/health` | Checks Qdrant and embedding readiness; reports Neo4j pending. |
| `POST` | `/query` | Dense retrieval followed by grounded Sarvam synthesis. |
| `POST` | `/ingest` | Returns the latest GPU ingestion summary; does not start a job. |

## Failure behavior

Embedding, Qdrant, and Sarvam failures are translated into safe 503 domain errors. A missing ingestion summary returns 404. Every API error follows `{ "error": { "code", "message", "details", "request_id" } }`.
