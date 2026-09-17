"""Typed application configuration loaded from the environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import AnyHttpUrl, Field, SecretStr, model_validator
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
    llm_synth_max_tokens: int = Field(default=4096, gt=0)
    llm_request_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=3, gt=0, le=10)
    ingestion_data_dir: Path = Path("data/raw")
    ingestion_summary_path: Path = Path("data/processed/last_ingestion.json")
    docling_vlm_url: AnyHttpUrl | None = None
    docling_vlm_token: SecretStr | None = None
    docling_vlm_model: str | None = None
    conversation_store: Literal["memory", "sqlite"] = "memory"
    conversation_history_max_turns: int = Field(default=20, gt=0)
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

    @property
    def cors_origins(self) -> list[str]:
        """Return configured CORS origins, ignoring blank comma-separated values."""
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]
