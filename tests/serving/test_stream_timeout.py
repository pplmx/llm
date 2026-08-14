"""Router-level idle timeout on the streaming /generate route.

RIL sweep finding (LOW): the streaming generator held ``inference_semaphore``
for the whole stream with no deadline, so a stalled/misbehaving model kept a
concurrency slot forever — with ``max_concurrent_requests`` stuck streams the
whole API would stop admitting new requests. The stream now applies an IDLE
timeout (``config.request_timeout`` with no chunk produced): a stream that
keeps producing tokens is never cut off; a stale one aborts and releases the
slot.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from prometheus_client import CollectorRegistry
from pydantic import ValidationError

import llm.serving.routers.chat as chat_module
import llm.serving.routers.generate as generate_module
from llm.serving.config import ServingConfig
from llm.serving.metrics import ServingMetrics
from llm.serving.schemas import ChatCompletionRequest, ChatMessage, GenerationRequest


@pytest.mark.quick
def test_stream_generator_stalled_stream_times_out_and_releases_slot(monkeypatch):
    """A stream that stops yielding for longer than request_timeout must
    abort with a 504-style error chunk and free the inference slot."""

    def _stalled():
        yield "first"
        time.sleep(30)  # simulate a model that stalled mid-generation

    monkeypatch.setattr(generate_module, "config", ServingConfig(request_timeout=0.15))
    monkeypatch.setattr(generate_module, "_sync_stream_generate", lambda **kw: _stalled())
    sem = asyncio.Semaphore(1)
    monkeypatch.setattr(generate_module, "inference_semaphore", sem)

    async def _collect():
        return [
            chunk async for chunk in generate_module._stream_generator(GenerationRequest(prompt="x", max_new_tokens=5))
        ]

    start = time.monotonic()
    chunks = asyncio.run(_collect())
    elapsed = time.monotonic() - start

    assert chunks[0] == "first"
    assert any("timed out" in c for c in chunks[1:]), chunks
    assert elapsed < 5.0, f"stream did not abort on idle timeout (took {elapsed:.1f}s)"
    assert not sem.locked(), "the inference slot must be released after the stream abort"


@pytest.mark.quick
def test_stream_generator_steady_stream_is_never_cut_off(monkeypatch):
    """A stream that keeps producing chunks (each faster than the idle window)
    must run to completion even past the total request_timeout budget."""

    def _steady():
        yield from ["a", "b", "c"]

    monkeypatch.setattr(generate_module, "config", ServingConfig(request_timeout=0.1))
    monkeypatch.setattr(generate_module, "_sync_stream_generate", lambda **kw: _steady())
    monkeypatch.setattr(generate_module, "inference_semaphore", None)

    async def _collect():
        return [
            chunk async for chunk in generate_module._stream_generator(GenerationRequest(prompt="x", max_new_tokens=5))
        ]

    assert asyncio.run(_collect()) == ["a", "b", "c"]


@pytest.mark.quick
def test_stream_generator_uses_config_timeout():
    """A non-positive request_timeout must be rejected at config time (it is
    the idle budget for streaming too)."""
    with pytest.raises(ValidationError):
        ServingConfig(request_timeout=0.0)


# --- Chat streaming idle timeout -----------------------------------------
#
# RIL round-45 hardening added an idle timeout to `/generate?stream=true`
# but not `/v1/chat/completions?stream=true`, which held the inference
# semaphore for the stream lifetime with the same stall profile. These
# tests pin the same behaviour for the chat route.


def _chat_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")], max_tokens=5)


@pytest.mark.quick
def test_chat_stream_stalled_stream_times_out_and_releases_slot(monkeypatch):
    """A chat stream that stops yielding for longer than request_timeout must
    abort with an SSE error chunk + [DONE] and free the inference slot."""

    def _stalled():
        yield "first"
        time.sleep(30)  # simulate a model that stalled mid-generation

    monkeypatch.setattr(chat_module, "config", ServingConfig(request_timeout=0.15))
    monkeypatch.setattr(chat_module, "_sync_stream_generate", lambda **kw: _stalled())
    sem = asyncio.Semaphore(1)
    monkeypatch.setattr(chat_module, "inference_semaphore", sem)

    async def _collect():
        return [
            chunk
            async for chunk in chat_module._chat_stream_generator(
                _chat_request(),
                "hi",
                repetition_penalty=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                logit_bias=None,
            )
        ]

    start = time.monotonic()
    chunks = asyncio.run(_collect())
    elapsed = time.monotonic() - start

    assert chunks[0].startswith("data: "), "first chunk is the SSE role chunk"
    assert any("data:" in c and "timed out" in c for c in chunks[1:]), chunks
    assert chunks[-1] == "data: [DONE]\n\n", chunks
    assert elapsed < 5.0, f"chat stream did not abort on idle timeout (took {elapsed:.1f}s)"
    assert not sem.locked(), "the inference slot must be released after the chat stream abort"


@pytest.mark.quick
def test_chat_stream_steady_stream_is_never_cut_off(monkeypatch):
    """A chat stream that keeps producing chunks faster than the idle window
    must run to completion even past the total request_timeout budget."""

    def _steady():
        yield from ["a", "b", "c"]

    monkeypatch.setattr(chat_module, "config", ServingConfig(request_timeout=0.1))
    monkeypatch.setattr(chat_module, "_sync_stream_generate", lambda **kw: _steady())
    monkeypatch.setattr(chat_module, "inference_semaphore", None)

    async def _collect():
        return [
            chunk
            async for chunk in chat_module._chat_stream_generator(
                _chat_request(),
                "hi",
                repetition_penalty=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                logit_bias=None,
            )
        ]

    chunks = asyncio.run(_collect())
    assert chunks[0].startswith("data: ")
    assert "a" in chunks[1], chunks
    assert "b" in chunks[2], chunks
    assert "c" in chunks[3], chunks
    assert chunks[-1] == "data: [DONE]\n\n", chunks


# --- Generate stream: timer status on timeout ----------------------------


def _histogram_count_and_sum(hist, **labels):
    """Return ``(count, sum)`` for a labelled histogram from ``.collect()``."""
    labelled = hist.labels(**labels)
    count = total = 0.0
    for metric in labelled.collect():
        for sample in metric.samples:
            if sample.name.endswith("_count"):
                count = int(sample.value)
            elif sample.name.endswith("_sum"):
                total = float(sample.value)
    return count, total


@pytest.mark.quick
def test_generate_stream_timeout_records_504_status(monkeypatch):
    """A timed-out stream must be recorded in the request_duration histogram
    under status=504, not clobbered to 200 by the happy-path footer."""

    def _stalled():
        yield "first"
        time.sleep(30)

    registry = CollectorRegistry()
    met = ServingMetrics(registry=registry)
    monkeypatch.setattr(generate_module, "metrics", met)
    monkeypatch.setattr(generate_module, "config", ServingConfig(request_timeout=0.15))
    monkeypatch.setattr(generate_module, "_sync_stream_generate", lambda **kw: _stalled())
    monkeypatch.setattr(generate_module, "inference_semaphore", None)

    async def _collect():
        return [
            chunk async for chunk in generate_module._stream_generator(GenerationRequest(prompt="x", max_new_tokens=5))
        ]

    asyncio.run(_collect())

    four, _ = _histogram_count_and_sum(met.request_duration_seconds, endpoint="generate", status="504")
    two, _ = _histogram_count_and_sum(met.request_duration_seconds, endpoint="generate", status="200")
    assert four == 1, f"timed-out stream must be recorded as 504, got 504-count={four}"
    assert two == 0, f"timed-out stream must not be recorded as 200, got 200-count={two}"
