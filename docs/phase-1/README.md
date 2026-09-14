# Phase 1 — Naive Text RAG

## Overview

Phase 1 provides the working safety-net pipeline: PDF/DOCX/Markdown narrative text is parsed on the GPU VM, chunked by SOP step, embedded with BGE-M3, stored in Qdrant Cloud, retrieved with dense similarity, and answered by Sarvam-105B with strict `[n]` citations. Any uncited generated claim is removed before the API responds.

## Contents

- [Architecture](./architecture.md)
- [Backend implementation](./backend.md)
- [Environment configuration](./environment.md)

## Quick start

### 1. CPU query API

From `backend/`:

```powershell
uv sync
Copy-Item .env.example .env
# Fill .env, including QDRANT_*, SARVAM_API_KEY, and EMBEDDING_SERVICE_URL.
# Optional keys may be left out entirely; do not leave them present but blank
# only when you want a non-default value. Blank optional keys fall back to
# their defaults automatically.
uv run python -m uvicorn src.api.main:create_app --factory --host 127.0.0.1 --port 8000
```

> Run all commands from `backend/`. The `python -m` form puts the project root
> on the import path so `src` and `scripts` resolve without extra setup.

### 2. GPU embedding service

On the Ubuntu GPU VM, from `backend/`:

```bash
uv sync --group gpu
uv run python -m uvicorn src.indexing.embedding_service:create_embedding_service_app \
  --factory --host 0.0.0.0 --port 8081
```

Keep port 8081 private and set the same `EMBEDDING_SERVICE_TOKEN` on both machines.

### 3. Ingest documents

Place `.pdf`, `.docx`, or `.md` files in `backend/data/raw/`, then run on the GPU VM (Markdown chunks have no page numbers):

```bash
uv run python -m scripts.run_ingestion --doc-type SOP --sop-version v1.0
```

For product documents, use `--doc-type product_info`. A run summary is written to `data/processed/last_ingestion.json`. Because ingestion and the API run on different machines, configure `INGESTION_SUMMARY_PATH` as the same shared mounted path on both hosts, or copy the completed JSON file to the API host before calling `POST /ingest`.

### 4. Query

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/query `
  -ContentType "application/json" `
  -Body '{"question":"Who approves large expenses?","conversation_id":null}'
```

## Validation

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pyrefly check
uv run pytest
```

Ruff and Pyrefly are clean and the unit suite passes. Live ingestion and question checks still require real documents, Qdrant credentials, a Sarvam key, and the GPU embedding service.

## Tech stack

| Concern | Technology |
|---|---|
| Parsing | Docling |
| Dense + sparse embeddings | BGE-M3 / FlagEmbedding |
| Query-time embedding transport | Private FastAPI service + HTTPX |
| Vector storage | Qdrant Cloud |
| Answer generation | Sarvam-105B |
| API | FastAPI |
| Validation | pytest, Ruff, Pyrefly |

## Verified references

- [Docling `DocumentConverter`](https://docling-project-docling.mintlify.app/api/document-converter)
- [Docling document and provenance model](https://docling-project-docling.mintlify.app/concepts/docling-document)
- [FlagEmbedding official repository](https://github.com/FlagOpen/FlagEmbedding)
- [Qdrant search documentation](https://qdrant.tech/documentation/search/)
- [Sarvam chat completions](https://docs.sarvam.ai/api-reference-docs/chat/chat-completions)

Content derived from external documentation was rephrased for compliance with licensing restrictions.
