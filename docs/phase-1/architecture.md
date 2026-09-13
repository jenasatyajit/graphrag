# Phase 1 Architecture

## Data flow

```text
GPU VM ingestion
PDF/DOCX/MD -> Docling text items -> SOP step chunker -> BGE-M3 dense+sparse
                                                      |
                                                      v
                                                Qdrant Cloud
                                                      ^
                                                      |
CPU query API                                         |
question -> private embedding HTTP service -> dense query
                                                -> numbered context
                                                -> Sarvam-105B
                                                -> citation validator
                                                -> POST /query response
```

The sparse vectors are generated and stored in Phase 1 even though retrieval is dense-only. Phase 2 can therefore enable hybrid fusion without changing the embedding contract or reworking ingestion.

## Key decisions

| Decision | Reason |
|---|---|
| GPU dependencies are lazily imported | The CPU query API must not import Docling, FlagEmbedding, or torch. |
| One embedding response contains dense and sparse vectors | Prevents a Phase 2 interface break and preserves BGE-M3 lexical output. |
| Ingestion runs on the GPU VM | Avoids transferring every document and removes an unnecessary HTTP embedding hop during batch work. |
| `POST /ingest` returns status only | The CPU API does not orchestrate jobs on the GPU VM. |
| Qdrant stores the complete chunk payload | Every answer citation remains traceable to document, page, and SOP version. |
| Citation filtering runs after generation | Unsupported or malformed generated claims cannot reach the caller. |

## Security boundaries

- The embedding service supports a bearer token and should be reachable only through a private network/VPN or protected reverse proxy.
- Qdrant and Sarvam credentials remain server-side in `.env`; they are never included in API responses or logs.
- Logs contain chunk IDs and counts, not complete document text.
- The API returns safe domain errors with a request ID instead of raw vendor exceptions.

## Health semantics

`GET /health` checks Qdrant, the query embedding service, and the current Neo4j placeholder. Qdrant and embeddings are required in Phase 1; if either is unavailable, the endpoint returns HTTP 503. Neo4j remains `pending` until Phase 3, so an otherwise usable Phase 1 deployment reports `degraded` with HTTP 200.
