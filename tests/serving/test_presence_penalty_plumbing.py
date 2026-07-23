"""Router-layer tests for presence_penalty plumbing (Tier 3 #37).

Verifies the chat router forwards ``request.presence_penalty`` as
its **own** kwarg to ``ServingGenerationService``, instead of folding
it into ``repetition_penalty`` like the pre-#37 implementation did.

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

    The lifespan normally loads a real model (which OOMs on
    CUDA-constrained boxes), so we mock `from_config` and
    `from_serving_config` to return lightweight mocks — keeping
    lifespan startup fast and memory-free. After the app starts we
    rebind the routers' module-level `generation_service` so the
    recording mock intercepts every request.
    """
    from llm.serving.batch_engine import ContinuousBatchingEngine
    from llm.serving.generation_service import ServingGenerationService

    mock = MagicMock()
    mock.generate.return_value = "ok"
    mock.stream.return_value = iter([])

    cfg = ServingConfig(
        api_key="test-key",
        request_timeout=30.0,
        chat_message_template="",
        chat_generation_prefix="",
        device="cpu",
    )

    # Prevent the real lifespan from loading a model — mock the two
    # factory calls it makes and the config-logging helper that pokes at
    # model internals.
    fake_service = MagicMock()
    fake_engine = MagicMock()
    monkeypatch.setattr(
        ServingGenerationService, "from_config",
        classmethod(lambda cls, config, **kw: fake_service),
    )
    monkeypatch.setattr(
        ContinuousBatchingEngine, "from_serving_config",
        classmethod(lambda cls, config, **kw: fake_engine),
    )
    monkeypatch.setattr(
        "llm.serving.api._log_server_config", lambda *a, **kw: None,
    )

    with TestClient(app) as c:
        # Rebind AFTER lifespan startup so the mock sticks for request time.
        monkeypatch.setattr(generate_module, "generation_service", mock)
        monkeypatch.setattr(generate_module, "config", cfg)
        monkeypatch.setattr(chat_module, "config", cfg)
        c.headers[api_key_header.model.name] = "test-key"
        yield c, mock

def test_chat_router_forwards_presence_penalty(client_with_mock):
    """`presence_penalty` from the chat request reaches the service as its own kwarg."""
    client, mock = client_with_mock
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "presence_penalty": 0.6,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200

    mock.generate.assert_called_once()
    kwargs = mock.generate.call_args.kwargs
    assert kwargs["presence_penalty"] == 0.6


def test_chat_router_does_not_alias_presence_into_repetition(client_with_mock):
    """The legacy ``1.0 + presence_penalty`` alias is gone.

    With ``presence_penalty=0.6``, ``repetition_penalty`` must stay
    at the OpenAI default of ``1.0`` — folding them together was the
    bug Tier 3 #37 fixed.
    """
    client, mock = client_with_mock
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "presence_penalty": 0.6,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200

    kwargs = mock.generate.call_args.kwargs
    assert kwargs["repetition_penalty"] == 1.0
    assert kwargs["presence_penalty"] == 0.6


def test_default_presence_penalty_is_zero(client_with_mock):
    """Omitting `presence_penalty` in the request defaults to 0 (no-op)."""
    client, mock = client_with_mock
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200

    kwargs = mock.generate.call_args.kwargs
    assert kwargs["presence_penalty"] == 0.0
    assert kwargs["repetition_penalty"] == 1.0


def test_chat_schema_no_longer_says_mapped_to_repetition():
    """The OpenAPI schema hint drops the '(mapped to repetition_penalty)' marker."""
    from llm.serving.schemas import ChatCompletionRequest

    field = ChatCompletionRequest.model_fields["presence_penalty"]
    assert "mapped to repetition_penalty" not in (field.description or "").lower()


def test_chat_schema_presence_penalty_range_is_openai_compatible():
    """`presence_penalty` schema range matches OpenAI's [-2.0, 2.0] spec."""
    from pydantic import ValidationError

    from llm.serving.schemas import ChatCompletionRequest

    # Inside the range → accepted.
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "presence_penalty": 2.0,
    }
    ChatCompletionRequest.model_validate(payload)

    payload["presence_penalty"] = -2.0
    ChatCompletionRequest.model_validate(payload)

    # Outside the range → rejected by the schema validator.
    payload["presence_penalty"] = 3.0
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(payload)
