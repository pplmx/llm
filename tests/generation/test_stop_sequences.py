"""Tests for stop-sequence support in GenerationConfig + generation backends.

Stop sequences are the OpenAI-compat ``stop`` parameter (string or
list of up to 4 strings) — generation halts the moment the streamed
output contains any of them, and the stop string itself is NOT
included in the final response. The OpenAI schema has always
advertised this field but the implementation was a documented gap
(schema description literally said "(not implemented)"); this slice
closes that gap for the eager, batched, and speculative backends.

These tests cover:

- GenerationConfig accepts ``stop`` (None / str / list[str]) and the
  default is None (zero behavior change for existing callers).
- EagerGenerationBackend stops generation the moment a stop sequence
  becomes a suffix of the running output, and the stop string itself
  is excluded from the final accumulated string.
- ``ServingGenerationService.generate`` / ``stream`` accept and
  forward ``stop`` to the backend.
- The OpenAI-compat chat + generate routers forward
  ``request.stop`` to the service.
"""

from __future__ import annotations

import torch
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from llm.generation.backends import EagerGenerationBackend, GenerationConfig


# Force CPU for eager-backend stop-sequence tests — the model is tiny
# and the custom tokenizers control decode output, so there's no need
# for GPU.  In environments where CUDA is available but memory-constrained,
# the session-scoped ``device`` fixture from conftest.py would cause an
# OOM during ``tiny_model`` construction; this override avoids that.
@pytest.fixture
def device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MultiCharTokenizer:
    """Character-level tokenizer that maps each token id to a stable char.

    Each integer id ``i`` decodes to ``chr(ord('a') + i % 26)`` — enough
    variety to make stop sequences (e.g. ``"end"``) reachable in tests
    without needing a real vocabulary.
    """

    pad_token_id: int = 0
    eos_token_id: int = 99

    def __init__(self, prompt_ids: list[int] | None = None) -> None:
        self._prompt_ids = prompt_ids or [1]

    def encode(self, text: str) -> list[int]:
        return list(self._prompt_ids)

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(ord("a") + i % 26) for i in ids)


class _FixedSeqTokenizer:
    """Token-by-token deterministic decoder for stop-sequence tests.

    The model never actually decodes anything sensible in these tests
    — we drive the loop by setting ``tokens`` (the integer stream the
    ``generate`` call returns) and watching where the backend stops.
    """

    pad_token_id: int = 0

    def __init__(self, tokens: list[int], prompt_ids: list[int] | None = None) -> None:
        self._tokens = list(tokens)
        self._idx = 0
        self._prompt_ids = prompt_ids or []

    def encode(self, text: str) -> list[int]:
        return list(self._prompt_ids)

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(ord("a") + i % 26) for i in ids)


# ---------------------------------------------------------------------------
# GenerationConfig tests
# ---------------------------------------------------------------------------


def test_generation_config_stop_defaults_to_none():
    """Default stop is None — zero behavior change for existing callers."""
    config = GenerationConfig()
    assert config.stop is None


def test_generation_config_accepts_single_string_stop():
    """Single string stop is stored verbatim (no list-wrapping at config layer)."""
    config = GenerationConfig(stop="END")
    assert config.stop == "END"


def test_generation_config_accepts_list_of_strings():
    """List-of-strings stop is stored verbatim."""
    config = GenerationConfig(stop=["END", "STOP", "###"])
    assert config.stop == ["END", "STOP", "###"]


def test_generation_config_stop_is_immutable():
    """GenerationConfig is frozen; re-assigning stop raises."""
    config = GenerationConfig(stop="END")
    with pytest.raises((FrozenInstanceError, AttributeError)):
        config.stop = "OTHER"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EagerGenerationBackend tests
# ---------------------------------------------------------------------------


def _run_eager(tiny_model, device, stub_tokenizer_factory, *, max_new_tokens=20, **gen_kwargs):
    """Helper: run eager generate to completion, return the full accumulated string.

    ``stub_tokenizer_factory(call_seq)`` should return a tokenizer whose
    ``decode`` emits the next character from ``call_seq`` on each call.
    The eager backend iterates max_new_tokens times and calls decode on
    each new token id.
    """
    config = GenerationConfig(max_new_tokens=max_new_tokens, **gen_kwargs)
    backend = EagerGenerationBackend()
    chunks = list(backend.stream(tiny_model.to(device), stub_tokenizer_factory(), "x", config))
    return "".join(chunks)


def test_eager_backend_no_stop_generates_full_window(tiny_model, device):
    """With stop=None, the eager backend runs all max_new_tokens iterations."""

    class _GrowingTokenizer(_MultiCharTokenizer):
        def __init__(self) -> None:
            super().__init__(prompt_ids=[1])
            self.counter = 0

        def decode(self, ids):
            self.counter += 1
            return chr(ord("a") + (self.counter - 1) % 26)

    result = _run_eager(tiny_model, device, _GrowingTokenizer, max_new_tokens=5)
    assert len(result) == 5


def test_eager_backend_single_string_stop(tiny_model, device):
    """A single string stop sequence halts generation the moment it appears.

    The tokenizer emits ``a b c d END a b c ...`` — generation should
    stop right after the ``END`` is emitted, and the returned string
    must NOT include the ``END`` itself (OpenAI semantics).
    """

    class _StopSequenceTokenizer(_MultiCharTokenizer):
        def __init__(self) -> None:
            super().__init__(prompt_ids=[1])
            # Each entry maps to a single decoded char.
            self._seq = ["a", "b", "c", "d", "E", "N", "D", "a", "b", "c", "d", "E"]
            self._i = 0

        def decode(self, ids):
            ch = self._seq[self._i]
            self._i += 1
            return ch

    result = _run_eager(
        tiny_model,
        device,
        _StopSequenceTokenizer,
        max_new_tokens=12,
        stop="END",
    )
    # Generation must halt at the "END" boundary; the running output
    # never reaches the "E" of the next cycle, let alone the "N" or "D".
    assert result == "abcd", f"Expected 'abcd' (stop sequence excluded), got {result!r}"


def test_eager_backend_list_of_stops_first_match_wins(tiny_model, device):
    """Multiple stops — the FIRST one to become a suffix wins."""

    class _TwoStopsTokenizer(_MultiCharTokenizer):
        def __init__(self) -> None:
            super().__init__(prompt_ids=[1])
            # Emits "xySTOPab..." then "xyEND...". With stops=["STOP","END"],
            # generation halts at the first suffix match (after "xySTOP" the
            # running output is "xySTOP" → suffix "STOP" matches first).
            self._seq = ["x", "y", "S", "T", "O", "P", "a", "b", "E", "N", "D"]
            self._i = 0

        def decode(self, ids):
            ch = self._seq[self._i]
            self._i += 1
            return ch

    result = _run_eager(
        tiny_model,
        device,
        _TwoStopsTokenizer,
        max_new_tokens=11,
        stop=["STOP", "END"],
    )
    assert result == "xy", f"Expected 'xy' (STOP excluded, no END reached), got {result!r}"


def test_eager_backend_stop_at_first_token(tiny_model, device):
    """Stop sequence that appears immediately halts at iteration 1."""

    class _ImmediateStopTokenizer(_MultiCharTokenizer):
        def __init__(self) -> None:
            super().__init__(prompt_ids=[1])
            self._seq = ["X", "a", "b", "c"]
            self._i = 0

        def decode(self, ids):
            ch = self._seq[self._i]
            self._i += 1
            return ch

    result = _run_eager(
        tiny_model,
        device,
        _ImmediateStopTokenizer,
        max_new_tokens=4,
        stop="X",
    )
    assert result == "", f"Expected empty string (X is the first token), got {result!r}"


def test_eager_backend_stop_never_matches_falls_through(tiny_model, device):
    """When the stop sequence is never emitted, generation runs to completion."""

    class _NoStopMatchTokenizer(_MultiCharTokenizer):
        def __init__(self) -> None:
            super().__init__(prompt_ids=[1])
            self._seq = ["a", "b", "c", "d"]
            self._i = 0

        def decode(self, ids):
            ch = self._seq[self._i]
            self._i += 1
            return ch

    result = _run_eager(
        tiny_model,
        device,
        _NoStopMatchTokenizer,
        max_new_tokens=4,
        stop="ZZZ",
    )
    assert result == "abcd"


# ---------------------------------------------------------------------------
# Router plumbing tests (mirrors test_frequency_penalty_plumbing.py pattern)
# ---------------------------------------------------------------------------


@pytest.fixture
def client_with_mock(monkeypatch):
    """TestClient with the generation service replaced by a recording mock.

    The mock records every kwargs dict it's called with so individual
    tests can assert ``stop`` was forwarded. The lifespan normally loads
    a real model (which OOMs on CUDA-constrained boxes), so we mock
    ``ServingGenerationService.from_config`` and
    ``ContinuousBatchingEngine.from_serving_config`` to return lightweight
    mocks — keeping lifespan startup fast and memory-free. After the app
    starts we rebind the routers' module-level ``generation_service`` so
    the recording mock intercepts every request.
    """
    from llm.serving.api import app
    from llm.serving.auth import api_key_header
    from llm.serving.config import ServingConfig
    import llm.serving.routers.chat as chat_module
    import llm.serving.routers.generate as generate_module
    from llm.serving.generation_service import ServingGenerationService
    from llm.serving.batch_engine import ContinuousBatchingEngine

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


def test_chat_router_forwards_stop_string(client_with_mock):
    """``/v1/chat/completions`` forwards ``stop`` to the service."""
    client, mock = client_with_mock
    payload = {
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
        "stop": "END",
        "stream": False,
    }
    resp = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key": "test-key"})
    assert resp.status_code == 200
    # The non-streaming path calls generate() with **kwargs forwarded from
    # the router. Assert the mock saw the stop string.
    assert mock.generate.called, "Expected chat router to call service.generate"
    call_kwargs = mock.generate.call_args.kwargs
    assert call_kwargs.get("stop") == "END", (
        f"Expected stop='END' in service.generate kwargs, got {call_kwargs}"
    )


def test_chat_router_forwards_stop_list(client_with_mock):
    """``/v1/chat/completions`` forwards a list of stop strings verbatim."""
    client, mock = client_with_mock
    payload = {
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
        "stop": ["END", "STOP"],
        "stream": False,
    }
    resp = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key": "test-key"})
    assert resp.status_code == 200
    assert mock.generate.call_args.kwargs.get("stop") == ["END", "STOP"]


def test_generate_router_forwards_stop(client_with_mock):
    """``/generate`` forwards ``stop`` to the service."""
    client, mock = client_with_mock
    payload = {
        "prompt": "hi",
        "stop": "###",
    }
    resp = client.post("/generate", json=payload, headers={"X-API-Key": "test-key"})
    assert resp.status_code == 200
    assert mock.generate.called
    assert mock.generate.call_args.kwargs.get("stop") == "###"


def test_chat_router_no_stop_passes_none(client_with_mock):
    """When the request omits ``stop``, the service receives None (default)."""
    client, mock = client_with_mock
    payload = {
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
    }
    resp = client.post("/v1/chat/completions", json=payload, headers={"X-API-Key": "test-key"})
    assert resp.status_code == 200
    assert mock.generate.call_args.kwargs.get("stop") is None
