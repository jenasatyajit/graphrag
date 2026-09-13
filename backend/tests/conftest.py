"""Shared fixtures for deterministic Phase 0 API tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, TypedDict

import pytest
from fastapi.testclient import TestClient

from src.api.config import Settings
from src.api.main import create_app


class TestSettingsValues(TypedDict):
    """Required Settings constructor values used by unit tests."""

    sarvam_api_key: str
    cohere_api_key: str
    exa_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    app_environment: Literal["test"]


def settings_values() -> TestSettingsValues:
    """Return non-secret placeholder values valid for local unit tests."""
    return {
        "sarvam_api_key": "test-sarvam-key",
        "cohere_api_key": "test-cohere-key",
        "exa_api_key": "test-exa-key",
        "qdrant_url": "https://qdrant.example.test",
        "qdrant_api_key": "test-qdrant-key",
        "neo4j_uri": "neo4j+s://neo4j.example.test",
        "neo4j_user": "test-user",
        "neo4j_password": "test-password",
        "app_environment": "test",
    }


@pytest.fixture
def settings() -> Settings:
    """Create test-only settings without reading a developer .env file."""
    return Settings(_env_file=None, **settings_values())


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """Create a test client for the configured application factory."""
    with TestClient(create_app(settings)) as test_client:
        yield test_client
