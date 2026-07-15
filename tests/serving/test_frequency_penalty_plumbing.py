"""Router-layer tests for frequency_penalty plumbing (Tier 3 #35).

Verifies the chat router actually forwards
``request.frequency_penalty`` through to the underlying
``ServingGenerationService``, instead of silently dropping it like
the pre-#35 implementation did.

The mock is installed AFTER TestClient's lifespan startup so the
lifespan's real-service ``configure()`` doesn't overwrite it. We
also explicitly turn off the lifespan startup since we don't need
a real model — the routers only consult their module-level
``generation_service`` attribute at request time.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import llm.serving.routers.chat as chat_module
import llm.serving.routers.generate as generate_module
from llm.serving.api import app
from llm.serving.auth import api_key_header
from llm.serving.config import ServingConfig


@pytest.fixture
def client_with_mock(monkeypatch):
    """TestClient with the generation service replaced by a recording mock.

    Skips lifespan startup (``raise_server_exceptions=False`` keeps
    the mock alive across the test) and rebinds the module-level
    ``generation_service`` after the app is built so the lifespan
    ``configure()`` can't override it.
    """
    mock = MagicMock()
    mock.generate.return_value = "ok"
    mock.stream.return_value = iter([])

    cfg = ServingConfig(
        api_key="test-key",
        request_timeout=30.0,
        chat_message_template="",
        chat_generation_prefix="",
    )

    with TestClient(app) as c:
        # Rebind AFTER lifespan startup so it sticks for request time.
        monkeypatch.setattr(generate_module, "generation_service", mock)
        monkeypatch.setattr(chat_module, "config", cfg)
        monkeypatch.setattr(generate_module, "config", cfg)
        c.headers[api_key_header.model.name] = "test-key"
        yield c, mock


def test_chat_router_forwards_frequency_penalty(client_with_mock):
    """`frequency_penalty` from the chat request reaches the service."""
    client, mock = client_with_mock
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "frequency_penalty": 0.7,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200

    mock.generate.assert_called_once()
    kwargs = mock.generate.call_args.kwargs
    assert kwargs["frequency_penalty"] == 0.7


def test_generate_router_forwards_frequency_penalty(client_with_mock):
    """`frequency_penalty` from the /generate request reaches the service."""
    client, mock = client_with_mock
    payload = {
        "prompt": "hi",
        "max_new_tokens": 4,
        "frequency_penalty": 1.3,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200

    mock.generate.assert_called_once()
    kwargs = mock.generate.call_args.kwargs
    assert kwargs["frequency_penalty"] == 1.3


def test_default_frequency_penalty_is_zero(client_with_mock):
    """Omitting `frequency_penalty` in the request defaults to 0 (no-op)."""
    client, mock = client_with_mock
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200

    kwargs = mock.generate.call_args.kwargs
    assert kwargs["frequency_penalty"] == 0.0


def test_chat_schema_no_longer_says_not_implemented():
    """The OpenAPI schema hint drops the '(not implemented)' marker."""
    from llm.serving.schemas import ChatCompletionRequest

    field = ChatCompletionRequest.model_fields["frequency_penalty"]
    assert "not implemented" not in (field.description or "").lower()
