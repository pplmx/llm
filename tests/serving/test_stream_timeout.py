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
from pydantic import ValidationError

import llm.serving.routers.generate as generate_module
from llm.serving.config import ServingConfig
from llm.serving.schemas import GenerationRequest


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
