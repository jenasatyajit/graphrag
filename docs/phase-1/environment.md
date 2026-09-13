# Phase 1 Environment

Copy `backend/.env.example` to `backend/.env`; never commit the populated file.

## Required for the CPU query API

| Variable | Required | Description | Example |
|---|---:|---|---|
| `SARVAM_API_KEY` | Yes | Server-side Sarvam subscription key | `***` |
| `QDRANT_URL` | Yes | Qdrant Cloud HTTPS endpoint | `https://cluster.example.qdrant.io` |
| `QDRANT_API_KEY` | Yes | Qdrant Cloud API key | `***` |
| `EMBEDDING_SERVICE_URL` | Yes | Private GPU service base URL | `https://embed.internal:8081` |
| `EMBEDDING_SERVICE_TOKEN` | Yes | Shared private service token | `***` |
| `LLM_MODEL` | No | Sarvam synthesis model | `sarvam-105b` |
| `LLM_SYNTH_REASONING_EFFORT` | No | Final-answer reasoning level | `low` |
| `LLM_SYNTH_MAX_TOKENS` | No | Completion token cap | `4096` |
| `LLM_REQUEST_TIMEOUT_SECONDS` | No | Sarvam request timeout | `60` |
| `LLM_MAX_RETRIES` | No | Retryable Sarvam attempt limit | `3` |
| `RETRIEVAL_TOP_K` | No | Dense candidates sent to synthesis | `10` |
| `QDRANT_COLLECTION_NAME` | No | Internal chunk collection | `company_knowledge` |

The current shared `Settings` model also validates the reserved Cohere, Exa, and Neo4j keys, although those services become functional in later phases. Populate them with real deployment values before starting the API.

## Required on the GPU VM

| Variable | Required | Description | Example |
|---|---:|---|---|
| `EMBEDDING_MODEL` | No | Hugging Face model ID | `BAAI/bge-m3` |
| `EMBEDDING_DEVICE` | No | FlagEmbedding device | `cuda:0` |
| `EMBEDDING_USE_FP16` | No | Use half precision | `true` |
| `EMBEDDING_DIMENSION` | No | Dense vector width | `1024` |
| `EMBEDDING_BATCH_SIZE` | No | Maximum/request batch | `32` |
| `EMBEDDING_INFERENCE_TIMEOUT_SECONDS` | No | Maximum GPU inference time | `90` |
| `INGESTION_DATA_DIR` | No | Source document directory | `data/raw` |
| `INGESTION_SUMMARY_PATH` | No | Last-run JSON output | `data/processed/last_ingestion.json` |

The ingestion process also needs the same Qdrant endpoint/key as the CPU API.

## Network and secret handling

- Keep `SARVAM_API_KEY`, `QDRANT_API_KEY`, and service tokens in a secret manager or protected `.env` file.
- Do not expose the embedding port directly to the public internet.
- Restrict Qdrant keys to the intended cluster.
- Use TLS or a private encrypted tunnel between the API and GPU VM.
- Never include actual secret values in logs, bug reports, screenshots, or committed documentation.

## Local tooling note

This project requires `uv`. If PowerShell reports that `uv` is unknown, install it using the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/), restart the terminal, and run commands from `backend/`.
