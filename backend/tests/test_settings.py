"""Tests for typed application configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from src.api.config import Settings
from tests.conftest import settings_values


def test_settings_loads_complete_values() -> None:
    """Settings should preserve the configured current-model defaults."""
    settings = Settings(_env_file=None, **settings_values())

    assert settings.llm_model == "sarvam-105b"
    assert settings.rerank_model == "cohere/rerank-4-fast"
    assert str(settings.openrouter_base_url).rstrip("/") == "https://openrouter.ai/api/v1"
    assert settings.cors_origins == []


def test_settings_fails_clearly_when_required_value_is_missing() -> None:
    """Missing service configuration should fail before the API starts."""
    with pytest.raises(PydanticValidationError, match="qdrant_url"):
        Settings(
            _env_file=None,
            sarvam_api_key="test-sarvam-key",
            openrouter_api_key="test-openrouter-key",
            exa_api_key="test-exa-key",
            qdrant_api_key="test-qdrant-key",
            neo4j_uri="neo4j+s://neo4j.example.test",
            neo4j_user="test-user",
            neo4j_password="test-password",
            app_environment="test",
        )


def test_blank_optional_values_fall_back_to_defaults() -> None:
    """Blank optional keys must use defaults instead of failing to parse."""
    settings = Settings(
        _env_file=None,
        **settings_values(),
        RERANK_TOP_N="",
        EMBEDDING_DIMENSION="",
        EMBEDDING_USE_FP16="",
        LLM_REQUEST_TIMEOUT_SECONDS="",
        CONVERSATION_STORE="",
        DOCLING_VLM_URL="",
    )

    assert settings.rerank_top_n == 5
    assert settings.embedding_dimension == 1024
    assert settings.embedding_use_fp16 is True
    assert settings.llm_request_timeout_seconds == 60.0
    assert settings.conversation_store == "memory"
    assert settings.docling_vlm_url is None


def test_blank_required_value_still_fails() -> None:
    """A blank required credential must still raise a clear error."""
    values = settings_values()
    values["qdrant_api_key"] = ""

    with pytest.raises(PydanticValidationError, match="qdrant_api_key"):
        Settings(_env_file=None, **values)


def test_settings_parses_cors_origins() -> None:
    """Comma-separated CORS origins should be normalized for middleware."""
    settings = Settings(
        _env_file=None,
        **settings_values(),
        cors_origins_raw="https://ui.example.test, https://admin.example.test",
    )

    assert settings.cors_origins == ["https://ui.example.test", "https://admin.example.test"]
