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
    assert settings.rerank_model == "rerank-v4.0-pro"
    assert settings.cors_origins == []


def test_settings_fails_clearly_when_required_value_is_missing() -> None:
    """Missing service configuration should fail before the API starts."""
    with pytest.raises(PydanticValidationError, match="qdrant_url"):
        Settings(
            _env_file=None,
            sarvam_api_key="test-sarvam-key",
            cohere_api_key="test-cohere-key",
            exa_api_key="test-exa-key",
            qdrant_api_key="test-qdrant-key",
            neo4j_uri="neo4j+s://neo4j.example.test",
            neo4j_user="test-user",
            neo4j_password="test-password",
            app_environment="test",
        )


def test_settings_parses_cors_origins() -> None:
    """Comma-separated CORS origins should be normalized for middleware."""
    settings = Settings(
        _env_file=None,
        **settings_values(),
        cors_origins_raw="https://ui.example.test, https://admin.example.test",
    )

    assert settings.cors_origins == ["https://ui.example.test", "https://admin.example.test"]
