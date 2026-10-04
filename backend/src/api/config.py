"""Typed application configuration loaded from the environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the API backend.

    Required service credentials are intentionally validated at startup instead of
    allowing an incompletely configured deployment to fail later during a request.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    sarvam_api_key: SecretStr = Field(min_length=1)
    openrouter_api_key: SecretStr = Field(min_length=1)
    exa_api_key: SecretStr = Field(min_length=1)
    qdrant_url: AnyHttpUrl
    qdrant_api_key: SecretStr = Field(min_length=1)
    neo4j_uri: str = Field(min_length=1)
    neo4j_user: str = Field(min_length=1)
    neo4j_password: SecretStr = Field(min_length=1)

    embedding_model: str = "BAAI/bge-m3"
    openrouter_base_url: AnyHttpUrl = AnyHttpUrl("https://openrouter.ai/api/v1")
    rerank_model: str = "cohere/rerank-4-fast"
    rerank_top_n: int = Field(default=5, gt=0)
    rerank_candidate_limit: int = Field(default=20, gt=0, le=1000)
    rerank_request_timeout_seconds: float = Field(default=15.0, gt=0)
    rerank_max_retries: int = Field(default=3, gt=0, le=10)
    llm_model: str = "sarvam-105b"
    llm_router_reasoning_effort: str | None = None
    llm_synth_reasoning_effort: str | None = "low"
    # Extraction reliably needs some reasoning budget on sarvam-105b: with it
    # disabled the model frequently returns empty replies for content-rich
    # chunks. "medium" produces a dense, correct graph; 4096 tokens leaves room
    # for reasoning tokens without truncating the JSON. Verified 2026-09-15.
    llm_extract_reasoning_effort: str | None = "medium"
    llm_extract_max_tokens: int = Field(default=8192, gt=0)
    max_agent_iterations: int = Field(default=3, gt=0)
    # Phase 5 router. Router/rewrite/direct-answer calls run with reasoning
    # disabled (llm_router_reasoning_effort=None) so a small budget suffices.
    llm_router_max_tokens: int = Field(default=1024, gt=0)
    # Minimum reranker relevance score for an internal chunk to count as
    # relevant evidence. Below it, the router guardrail escalates to web search.
    internal_relevance_threshold: float = Field(default=0.2, ge=0.0, le=1.0)

    # Phase 5 Exa live web search (results are per-request only, never stored).
    exa_search_type: Literal["auto", "fast", "instant"] = "auto"
    exa_num_results: int = Field(default=5, gt=0, le=25)
    exa_highlight_max_characters: int = Field(default=1500, gt=0)
    exa_request_timeout_seconds: float = Field(default=20.0, gt=0)
    exa_max_retries: int = Field(default=2, gt=0, le=10)

    neo4j_database: str = "neo4j"
    graph_query_timeout_seconds: float = Field(default=15.0, gt=0)
    graph_max_retries: int = Field(default=3, gt=0, le=10)
    graph_traversal_max_hops: int = Field(default=2, ge=1, le=3)
    graph_entity_match_limit: int = Field(default=10, gt=0, le=100)
    graph_fact_limit: int = Field(default=15, gt=0, le=100)

    embedding_service_url: AnyHttpUrl | None = None
    embedding_service_token: SecretStr | None = None
    embedding_dimension: int = Field(default=1024, gt=0)
    embedding_batch_size: int = Field(default=32, gt=0, le=256)
    embedding_request_timeout_seconds: float = Field(default=30.0, gt=0)
    embedding_inference_timeout_seconds: float = Field(default=90.0, gt=0)
    embedding_max_retries: int = Field(default=3, gt=0, le=10)
    embedding_device: str = "cuda:0"
    embedding_use_fp16: bool = True
    qdrant_collection_name: str = "company_knowledge"
    qdrant_dense_vector_name: str = "dense"
    qdrant_sparse_vector_name: str = "sparse"
    retrieval_top_k: int = Field(default=10, gt=0, le=100)
    # Synthesis runs with reasoning enabled (llm_synth_reasoning_effort), and on
    # sarvam-105b reasoning tokens are drawn from max_tokens BEFORE visible
    # content. 4096 was too low for multi-source synthesis prompts: reasoning
    # consumed the budget and content came back empty (finish_reason="length").
    # 8192 leaves room for both. Verified against Sarvam docs 2026-10-03.
    llm_synth_max_tokens: int = Field(default=8192, gt=0)
    llm_request_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=3, gt=0, le=10)
    ingestion_data_dir: Path = Path("data/raw")
    ingestion_summary_path: Path = Path("data/processed/last_ingestion.json")
    # Phase 6 evaluation (RAGAS + ablation). The judge reuses sarvam_api_key and
    # talks to Sarvam's OpenAI-compatible endpoint; AnswerRelevancy embeddings
    # reuse the query-time embedding service. These are read only by the offline
    # eval CLI (scripts/run_eval.py), never by the request path.
    eval_golden_set_path: Path = Path("data/eval/golden_set.jsonl")
    eval_output_dir: Path = Path("data/eval/reports")
    eval_judge_model: str = "sarvam-105b"
    eval_judge_base_url: AnyHttpUrl = AnyHttpUrl("https://api.sarvam.ai/v1")
    eval_judge_max_concurrency: int = Field(default=4, gt=0, le=32)
    eval_judge_timeout_seconds: float = Field(default=60.0, gt=0)
    # The judge is sarvam-105b, where reasoning tokens are drawn from max_tokens
    # BEFORE the visible verdict. RAGAS collection metrics ask for structured
    # JSON; with reasoning on, the budget is spent before the JSON finishes and
    # ragas reports "output is incomplete due to a max_tokens length limit".
    # Disabling reasoning (None) and allowing a generous budget fixes it.
    eval_judge_max_tokens: int = Field(default=4096, gt=0)
    eval_judge_reasoning_effort: str | None = None
    # Documents whether the live corpus contains table/image_caption chunks, so
    # the ablation runner can label the +graph+multimodal stage honestly (there
    # is no query-time switch that disables multimodal chunks).
    eval_corpus_includes_multimodal: bool = False
    docling_vlm_url: AnyHttpUrl | None = None
    docling_vlm_token: SecretStr | None = None
    docling_vlm_model: str | None = None
    # Phase 4 multimodal ingestion (GPU VM only). Picture captioning is enabled
    # only when docling_vlm_url is set; otherwise ingestion still extracts tables
    # and skips captions so the batch job degrades gracefully.
    docling_image_scale: float = Field(default=2.0, gt=0)
    docling_caption_prompt: str = (
        "Describe this image or chart factually for document retrieval. "
        "Be concise and specific about any data, labels, and values shown."
    )
    docling_caption_timeout_seconds: float = Field(default=90.0, gt=0)
    docling_extract_tables: bool = True
    docling_extract_charts: bool = True
    ingestion_image_dir: Path = Path("data/processed/images")
    conversation_store: Literal["memory", "sqlite"] = "memory"
    conversation_history_max_turns: int = Field(default=20, gt=0)
    conversation_db_path: Path = Path("data/processed/conversations.db")
    conversation_ttl_hours: float = Field(default=24.0, gt=0)
    # Prior turns shown to the follow-up rewriter and the synthesizer.
    conversation_context_turns: int = Field(default=3, gt=0, le=20)
    app_environment: Literal["development", "production", "test"] = "development"
    cors_origins_raw: str = Field(default="", validation_alias="CORS_ORIGINS")

    @model_validator(mode="before")
    @classmethod
    def _drop_blank_optional_values(cls, data: Any) -> Any:
        """Treat blank optional env values as unset so defaults apply.

        A common mistake is copying ``.env.example`` and leaving optional keys
        present but empty. An empty string cannot be parsed as an int/float/bool/
        URL, so drop blanks for any field that has a default and let its default
        take over. Required credential fields have no default, so a truly blank
        value still fails with a clear "field required" error.
        """
        if not isinstance(data, dict):
            return data

        optional_fields = {
            name for name, field in cls.model_fields.items() if not field.is_required()
        }
        aliases = {
            field.validation_alias: name
            for name, field in cls.model_fields.items()
            if isinstance(field.validation_alias, str)
        }

        cleaned: dict[Any, Any] = {}
        for key, value in data.items():
            field_name = aliases.get(key, key)
            canonical = field_name.lower() if isinstance(field_name, str) else field_name
            is_blank = isinstance(value, str) and value.strip() == ""
            if is_blank and canonical in optional_fields:
                continue
            cleaned[key] = value
        return cleaned

    @field_validator(
        "llm_router_reasoning_effort",
        "llm_synth_reasoning_effort",
        "llm_extract_reasoning_effort",
        "eval_judge_reasoning_effort",
        mode="before",
    )
    @classmethod
    def _null_reasoning_effort(cls, value: Any) -> Any:
        """Map the literal strings "null"/"none" (as written in SPEC 5) to ``None``.

        Sarvam disables reasoning only for a real ``None``; forwarding the string
        "null" would be rejected by the API.
        """
        if isinstance(value, str) and value.strip().lower() in {"null", "none"}:
            return None
        return value

    @property
    def cors_origins(self) -> list[str]:
        """Return configured CORS origins, ignoring blank comma-separated values."""
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]
