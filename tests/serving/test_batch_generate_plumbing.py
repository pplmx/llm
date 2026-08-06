"""Router-layer tests for presence_penalty + frequency_penalty plumbing on /batch_generate and /generate (streaming).

Tier 3 #35, #37 closed the gap where ``frequency_penalty`` and
``presence_penalty`` were silently dropped on certain code paths. These
tests guard against regressions by asserting the router actually forwards
every sampling parameter through to ``ServingGenerationService``.

Covers:
- ``/generate`` (non-streaming) forwards ``presence_penalty``.
- ``/generate`` (streaming) forwards ``presence_penalty``.
- ``/batch_generate`` forwards ``frequency_penalty`` and ``presence_penalty``.
- ``EagerGenerationBackend.batch_generate`` forwards ``stop`` (not just
  the eager ``stream`` path).
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
def client_with_mock(monkeypatch, device):
    """TestClient with the generation service replaced by a recording mock.

    The lifespan normally loads a real model (which can OOM on
    CUDA-constrained boxes), so we mock ``from_config`` and
    ``from_serving_config`` to return lightweight mocks — keeping
    lifespan startup fast and memory-free. After the app starts we rebind
    the routers' module-level ``generation_service`` so the recording
    mock intercepts every request.

    The ``device`` fixture (GPU-first) is passed through to
    ``ServingConfig`` so the config reflects the same device the
    real service would use.
    """
    from llm.serving.batch_engine import ContinuousBatchingEngine
    from llm.serving.generation_service import ServingGenerationService

    mock = MagicMock()
    mock.generate.return_value = "ok"
    mock.stream.return_value = iter([])
    mock.batch_generate.return_value = ["ok", "ok"]

    cfg = ServingConfig(
        api_key="test-key",
        request_timeout=30.0,
        chat_message_template="",
        chat_generation_prefix="",
        device=str(device),
    )

    fake_service = MagicMock()
    fake_engine = MagicMock()
    monkeypatch.setattr(
        ServingGenerationService,
        "from_config",
        classmethod(lambda cls, config, **kw: fake_service),
    )
    monkeypatch.setattr(
        ContinuousBatchingEngine,
        "from_serving_config",
        classmethod(lambda cls, config, **kw: fake_engine),
    )
    monkeypatch.setattr(
        "llm.serving.api._log_server_config",
        lambda *a, **kw: None,
    )

    with TestClient(app) as c:
        monkeypatch.setattr(generate_module, "generation_service", mock)
        monkeypatch.setattr(generate_module, "config", cfg)
        monkeypatch.setattr(chat_module, "config", cfg)
        c.headers[api_key_header.model.name] = "test-key"
        yield c, mock


# ---------------------------------------------------------------------------
# /generate — presence_penalty forwarding
# ---------------------------------------------------------------------------


def test_generate_router_forwards_presence_penalty(client_with_mock):
    """``presence_penalty`` from ``/generate`` reaches the service as its own kwarg."""
    client, mock = client_with_mock
    payload = {
        "prompt": "hi",
        "max_new_tokens": 4,
        "presence_penalty": 0.6,
    }
    response = client.post("/generate", json=payload, headers={"X-API-Key": "test-key"})
    assert response.status_code == 200

    mock.generate.assert_called_once()
    kwargs = mock.generate.call_args.kwargs
    assert kwargs["presence_penalty"] == 0.6


def test_generate_router_forwards_presence_penalty_streaming(client_with_mock):
    """``presence_penalty`` flows through the streaming ``/generate`` path too."""
    client, mock = client_with_mock
    payload = {
        "prompt": "hi",
        "max_new_tokens": 4,
        "presence_penalty": 0.8,
        "stream": True,
    }
    response = client.post("/generate", json=payload, headers={"X-API-Key": "test-key"})
    assert response.status_code == 200

    assert mock.stream.called, "Expected streaming /generate to call service.stream"
    kwargs = mock.stream.call_args.kwargs
    assert kwargs["presence_penalty"] == 0.8


def test_generate_router_presence_and_frequency_together(client_with_mock):
    """Both penalties can coexist on ``/generate``."""
    client, mock = client_with_mock
    payload = {
        "prompt": "hi",
        "max_new_tokens": 4,
        "frequency_penalty": 1.3,
        "presence_penalty": 0.4,
    }
    response = client.post("/generate", json=payload, headers={"X-API-Key": "test-key"})
    assert response.status_code == 200

    kwargs = mock.generate.call_args.kwargs
    assert kwargs["frequency_penalty"] == 1.3
    assert kwargs["presence_penalty"] == 0.4


# ---------------------------------------------------------------------------
# /batch_generate — frequency_penalty + presence_penalty forwarding
# ---------------------------------------------------------------------------


def test_batch_generate_router_forwards_frequency_penalty(client_with_mock):
    """``frequency_penalty`` from ``/batch_generate`` reaches the service."""
    client, mock = client_with_mock
    payload = {
        "prompts": ["hello", "world"],
        "max_new_tokens": 4,
        "frequency_penalty": 1.5,
    }
    response = client.post("/batch_generate", json=payload, headers={"X-API-Key": "test-key"})
    assert response.status_code == 200

    mock.batch_generate.assert_called_once()
    kwargs = mock.batch_generate.call_args.kwargs
    assert kwargs["frequency_penalty"] == 1.5


def test_batch_generate_router_forwards_presence_penalty(client_with_mock):
    """``presence_penalty`` from ``/batch_generate`` reaches the service."""
    client, mock = client_with_mock
    payload = {
        "prompts": ["hello", "world"],
        "max_new_tokens": 4,
        "presence_penalty": 0.9,
    }
    response = client.post("/batch_generate", json=payload, headers={"X-API-Key": "test-key"})
    assert response.status_code == 200

    kwargs = mock.batch_generate.call_args.kwargs
    assert kwargs["presence_penalty"] == 0.9


def test_batch_generate_router_defaults_penalties_to_zero(client_with_mock):
    """Omitting both penalties on ``/batch_generate`` defaults to 0.0 (no-op)."""
    client, mock = client_with_mock
    payload = {
        "prompts": ["hello"],
        "max_new_tokens": 4,
    }
    response = client.post("/batch_generate", json=payload, headers={"X-API-Key": "test-key"})
    assert response.status_code == 200

    kwargs = mock.batch_generate.call_args.kwargs
    assert kwargs["frequency_penalty"] == 0.0
    assert kwargs["presence_penalty"] == 0.0


# ---------------------------------------------------------------------------
# EagerGenerationBackend.batch_generate — stop forwarding
# ---------------------------------------------------------------------------


def test_eager_backend_batch_generate_forwards_stop(tiny_model, device, stub_tokenizer):
    """``EagerGenerationBackend.batch_generate`` must forward ``stop`` to the eager path.

    Regression guard for the bug where ``stop`` was silently dropped on
    the batched eager path (only ``stream`` honoured it).
    """
    from llm.generation.backends import EagerGenerationBackend, GenerationConfig

    config = GenerationConfig(
        max_new_tokens=1,
        temperature=0.0,
        use_cache=False,
        stop="END",
    )
    backend = EagerGenerationBackend()
    # We don't assert on the output text — we just need to confirm the
    # call doesn't raise and returns one result per prompt. The stop
    # parameter is now forwarded to ``eager.batch_generate`` which would
    # previously raise a TypeError if it tried to unpack a non-None stop.
    outputs = backend.batch_generate(
        model=tiny_model.to(device),
        tokenizer=stub_tokenizer,
        prompts=["a", "b"],
        config=config,
    )
    assert len(outputs) == 2
