"""Router-level idle timeout on the streaming /generate route.

RIL sweep finding (LOW): the streaming generator held ``inference_semaphore``
for the whole stream with no deadline, so a stalled/misbehaving model kept a
concurrency slot forever — with ``max_concurrent_requests`` stuck streams the
whole API would stop admitting new requests. The stream now applies an IDLE
timeout (``config.request_timeout`` with no chunk produced): a stream that
keeps producing tokens is never cut off; a stale one aborts and releases the
slot.

F2 (TASK-109 / ISS-122): the router's idle timeout must not leave the
abandoned sync engine generator (``stream_request``) running — it holds the
KV/scheduler slot and burns forward passes until the async generator is GC'd.
The ``_drive_sync_iterator`` bridge closes the sync generator once the
in-flight step completes, so ``stream_request``'s ``finally`` reap runs
promptly. These tests pin that contract.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from prometheus_client import CollectorRegistry
from pydantic import ValidationError

import llm.serving.routers.chat as chat_module
import llm.serving.routers.generate as generate_module
from llm.serving.batch_engine import ContinuousBatchingEngine
from llm.serving.config import ServingConfig
from llm.serving.metrics import ServingMetrics
from llm.serving.schemas import ChatCompletionRequest, ChatMessage, GenerationRequest

# --- F2: abandoned sync generator is closed (not left for GC) ---------------


@pytest.mark.quick
def test_drive_sync_iterator_closes_generator_when_consumer_abandons():
    """RIL ISS-122/F2: abandoning the async bridge (idle timeout) must close
    the underlying sync generator after its in-flight step — NOT leave it
    suspended holding resources until asyncgen GC finalizes it."""

    closed: list[bool] = []

    def sync_gen():
        try:
            yield "a"
            time.sleep(0.3)  # the in-flight next() the timeout abandons
            yield "b"
        finally:
            closed.append(True)

    async def _drive():
        it = sync_gen()
        bridge = generate_module._drive_sync_iterator(it)
        assert await anext(bridge) == "a"
        # Simulate the route's idle timeout abandoning the second chunk
        # mid-step: the await is cancelled while the worker thread is still
        # inside next() (sleeping).
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await anext(bridge)

    asyncio.run(_drive())

    # The abandoned step still finishes on its own worker thread, then the
    # generator is closed (its finally-reap runs) — WITHOUT waiting for the
    # asyncgen to be GC'd. Poll synchronously: the close fires from the
    # worker-thread done-callback, independently of the now-closed loop.
    deadline = time.monotonic() + 2.0
    while not closed and time.monotonic() < deadline:
        time.sleep(0.05)
    assert closed, "sync generator must be closed after abandonment, not left for GC"


@pytest.mark.quick
def test_generate_stream_timeout_releases_engine_slot(tiny_model, device, stub_tokenizer):
    """F2 end-to-end: an idle-timeout-aborted /generate stream must release
    the engine's KV slot promptly (close of stream_request runs its finally
    reap), instead of pinning the slot until the asyncgen is GC'd.

    Reproduces ISS-122: the route streams through a REAL engine whose step is
    slow enough that the idle window fires mid-generation; after the stream
    aborts, the engine slot for the request must be freed without any manual
    gc.collect(). The reap runs from the worker-thread done-callback, so it
    must not depend on the event loop still being alive.
    """
    tiny_model.eval()
    engine = ContinuousBatchingEngine(
        model=tiny_model,
        tokenizer=stub_tokenizer,
        max_batch_size=2,
        device=str(device),
    )

    req = GenerationRequest(prompt="abcd", max_new_tokens=50)
    req.request_id = "idle-timeout-slot"

    real_step = engine.step

    def slow_step():
        time.sleep(0.4)  # every step is slow → the idle window fires mid-step
        return real_step()

    engine.step = slow_step  # type: ignore[method-assign]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(generate_module, "config", ServingConfig(request_timeout=0.15))
    monkeypatch.setattr(generate_module, "_sync_stream_generate", lambda **_kw: engine.stream_request(req))
    monkeypatch.setattr(generate_module, "inference_semaphore", None)

    async def _collect():
        return [
            chunk
            async for chunk in generate_module._stream_generator(GenerationRequest(prompt="abcd", max_new_tokens=50))
        ]

    try:
        chunks = asyncio.run(_collect())
        assert any("timed out" in c for c in chunks), chunks

        # The engine slot must be freed promptly (background reap), without
        # waiting for the asyncgen GC finalizer.
        deadline = time.monotonic() + 5.0
        while engine.slot_allocator.get_slot("idle-timeout-slot") >= 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        slot = engine.slot_allocator.get_slot("idle-timeout-slot")
        assert slot == -1, f"KV slot must be freed after stream timeout, still = {slot}"
    finally:
        monkeypatch.undo()


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
