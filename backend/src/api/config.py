"""Typed application configuration loaded from the environment."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr
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
    cohere_api_key: SecretStr = Field(min_length=1)
    exa_api_key: SecretStr = Field(min_length=1)
    qdrant_url: AnyHttpUrl
    qdrant_api_key: SecretStr = Field(min_length=1)
    neo4j_uri: str = Field(min_length=1)
    neo4j_user: str = Field(min_length=1)
    neo4j_password: SecretStr = Field(min_length=1)

    embedding_model: str = "BAAI/bge-m3"
    rerank_model: str = "rerank-v4.0-pro"
    rerank_top_n: int = Field(default=5, gt=0)
    llm_model: str = "sarvam-105b"
    llm_router_reasoning_effort: str | None = None
    llm_synth_reasoning_effort: str | None = "low"
    max_agent_iterations: int = Field(default=3, gt=0)

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

    @property
    def cors_origins(self) -> list[str]:
        """Return configured CORS origins, ignoring blank comma-separated values."""
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]
