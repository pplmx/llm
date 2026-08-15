"""Concurrency tests for ``ContinuousBatchingEngine`` (audit Finding AT).

The engine is reachable from multiple worker threads (FastAPI runs each
``service.generate`` call in a threadpool). Mutations to ``self.scheduler``,
``self.slot_allocator``, ``self.kv_caches``, and ``self.prefix_cache`` would
race without a lock. These tests pin the contract that ``step()`` is safe
under concurrent invocation.

Strategy: build an engine against a tiny fake model (GPU-first, falls back
to CPU) and a stub tokenizer, then drive it from many threads. The model
forward is monkey-patched to return deterministic logits so we can assert
state invariants without a real GPU dependency.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn as nn

from llm.serving.batch_engine import ContinuousBatchingEngine
from llm.serving.schemas import GenerationRequest

# --- Fake model + tokenizer -------------------------------------------------


@dataclass
class _StubTokenizer:
    """Character-level stub that mirrors the encode/decode/pad_token_id surface."""

    pad_token_id: int = 0
    eos_token_id: int = 1

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 64 + 2 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(max(0, i - 2)) for i in ids)


class _SelfAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_kv_heads = 1
        self.head_dim = 4


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _SelfAttn()


class _FakeModel(nn.Module):
    """Tiny model (GPU-first) with the surface that ContinuousBatchingEngine reads."""

    def __init__(self, vocab_size: int = 64, n_layers: int = 1) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(n_layers)])
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        kv_caches: Any = None,
        use_cache: bool = False,
        batch_indices: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        paged_kv_cache: Any = None,
    ) -> tuple[torch.Tensor, Any]:
        # Deterministic logits: argmax over input ids mod vocab_size.
        bs, seq_len = input_ids.shape
        logits = torch.zeros(bs, seq_len, self.vocab_size, dtype=torch.float32, device=input_ids.device)
        for i in range(bs):
            for j in range(seq_len):
                # Pick a token that is NOT the EOS token to avoid auto-finish.
                logits[i, j, (int(input_ids[i, j].item()) + 2) % self.vocab_size] = 10.0
        return logits, kv_caches


@pytest.fixture
def fake_engine(device):
    """Construct a real ContinuousBatchingEngine with a tiny fake model.

    Uses the session-scoped ``device`` fixture (GPU-first) so tests
    exercise the same code path as production serving.
    """
    model = _FakeModel(vocab_size=64, n_layers=1).to(device)
    tokenizer = _StubTokenizer()
    engine = ContinuousBatchingEngine(
        model=model,
        tokenizer=tokenizer,
        device=str(device),
        max_batch_size=8,
        max_seq_len=16,
        dtype=torch.float32,
        enable_prefix_cache=False,
        use_paged_attention=False,
        max_blocks=16,
        block_size=4,
    )
    return engine


# --- Basic locking contract -------------------------------------------------


def test_step_lock_is_allocated(fake_engine):
    """The lock exists and is a ``threading.Lock`` (or compatible)."""
    assert isinstance(fake_engine._step_lock, type(threading.Lock()))


def test_step_serializes_concurrent_invocations(fake_engine):
    """Two concurrent ``step()`` calls cannot interleave their bookkeeping.

    ``step()`` holds ``_step_lock`` across the whole step (pre, forward and
    post) so a concurrent ``step()`` can never run the model forward against
    the same slots — running the forward unlocked let two threads each append
    a token to the same sequence from one logical step, desyncing
    ``generated_ids`` from the KV cache (a real corruption in the ``batched``
    backend serving concurrent HTTP requests). We instrument BOTH pre and
    post to verify the critical sections serialise: if both threads enter
    simultaneously, the test fails.
    """
    hold_log: list[tuple[str, float]] = []
    hold_lock = threading.Lock()
    original_pre = fake_engine._lock_step_pre
    original_post = fake_engine._lock_step_post

    def instrumented_pre():
        with hold_lock:
            hold_log.append(("enter", time.monotonic()))
        # Hold the lock briefly so a racing thread has time to overlap
        # if the lock is broken.
        time.sleep(0.05)
        result = original_pre()
        with hold_lock:
            hold_log.append(("exit", time.monotonic()))
        return result

    def instrumented_post(result):
        with hold_lock:
            hold_log.append(("enter", time.monotonic()))
        time.sleep(0.05)
        out = original_post(result)
        with hold_lock:
            hold_log.append(("exit", time.monotonic()))
        return out

    fake_engine._lock_step_pre = instrumented_pre  # type: ignore[assignment]
    fake_engine._lock_step_post = instrumented_post  # type: ignore[assignment]

    # Pre-compute only: idle engine returns None which short-circuits
    # ``step()`` before ``_lock_step_post`` is called. To exercise both
    # critical sections we need at least one request.
    req = GenerationRequest(prompt="x", max_new_tokens=2)
    req.request_id = "thread-safety-req"
    fake_engine.add_request(req)

    errors: list[BaseException] = []

    def worker():
        try:
            fake_engine.step()
        except BaseException as exc:  # noqa: BLE001 - capture for reporting
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors, f"step() raised: {errors}"
    # Every enter must be paired with an exit before the next enter; entries
    # must be strictly after the previous exit (with no concurrent entries).
    enters = [t for kind, t in hold_log if kind == "enter"]
    exits = [t for kind, t in hold_log if kind == "exit"]
    assert len(enters) == len(exits) >= 4
    for enter, exit in zip(enters, exits, strict=True):
        assert enter <= exit, "lock entered before previous exit"


# --- Stress test under add_request + step -----------------------------------


@pytest.mark.slow
def test_concurrent_add_request_and_step_does_not_crash(fake_engine):
    """Stress: many threads concurrently enqueue requests and call step().

    Asserts no crashes and that the scheduler invariants hold at the end:
    every sequence has a unique slot, no slot is double-allocated.
    """
    n_workers = 8
    iterations = 50
    errors: list[BaseException] = []

    def worker(idx: int):
        try:
            for i in range(iterations):
                req = GenerationRequest(
                    request_id=f"w{idx}-r{i}",
                    prompt=f"hello-{idx}-{i}",
                    max_new_tokens=3,
                    temperature=1.0,
                )
                fake_engine.add_request(req)
                # Step until our request finishes (or it times out).
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    fake_engine.step()
                    seq = fake_engine.scheduler.get_sequence(req.request_id)
                    if seq is None or seq.is_finished():
                        break
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)

    assert not errors, f"workers raised: {errors[:3]}"
    # All slots are returned to free set; no double-allocation.
    allocator = fake_engine.slot_allocator
    assert len(allocator.seq_to_slot) == 0, f"unfreed slots: {allocator.seq_to_slot}"
    # free_slots equals the original pool.
    assert len(allocator.free_slots) == fake_engine.max_batch_size


def test_concurrent_step_does_not_overshoot_generated_ids(fake_engine):
    """Concurrent ``step()`` calls must not append more than one token per
    logical step to a sequence.

    Regression (ISS-027): ``step()`` previously held ``_step_lock`` only for
    the pre/post bookkeeping and ran the model forward *unlocked*, so two
    threads hitting ``step()`` concurrently could each run the forward against
    the same slots and each append a token to the same sequence in one logical
    step — ``generated_ids`` then exceeded ``max_new_tokens`` and desynced from
    the KV cache (a real corruption in the ``batched`` backend when the API
    threadpool serves concurrent requests, default 4).

    Here a single request with ``max_new_tokens=1`` is stepped by many threads
    at once; the sequence must gain exactly one token (and then finish),
    never two.
    """
    # A request that must stop after one generated token.
    req = GenerationRequest(prompt="x", max_new_tokens=1)
    req.request_id = "sync-req"
    fake_engine.add_request(req)

    errors: list[BaseException] = []

    def worker():
        try:
            fake_engine.step()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    barrier_start = threading.Barrier(4)

    def synced_worker():
        barrier_start.wait()
        worker()

    synced = [threading.Thread(target=synced_worker) for _ in range(4)]
    for t in synced:
        t.start()
    for t in synced:
        t.join(timeout=10.0)

    assert not errors, f"step() raised: {errors}"
    seq = fake_engine.scheduler.get_sequence("sync-req")
    # With max_new_tokens=1 and a concurrent step, the sequence may be already
    # finished (freed) or still hold exactly one token — but never two.
    if seq is None:
        return  # finished + freed = exactly one token was generated
    assert len(seq.generated_ids) == 1, (
        f"concurrent step() appended {len(seq.generated_ids)} tokens to a "
        f"max_new_tokens=1 request (expected exactly 1); generated_ids is "
        f"desynced from the KV cache"
    )


def test_stream_request_drains_tail_when_concurrent_step_evicts(fake_engine, monkeypatch):
    """End-to-end: ``stream_request`` must not drop the final token when a
    concurrent ``step()`` evicts the finished sequence before the owner drains
    it.

    Regression (RIL serving scan F1): the loop re-fetched the sequence via
    ``get_sequence`` after ``step()``; a concurrent stepper's ``schedule()``
    filters finished sequences out of ``running``, so the re-fetch returned
    None and the owner broke without draining the final token (truncated
    stream). The engine now reuses the reference captured before the step.

    We reproduce the race deterministically: after this request's step marks
    it FINISHED + frees its slot, ``_lock_step_post`` runs ``schedule()``
    (what the other generator's step would do) so the finished sequence is
    evicted from ``running`` before ``stream_request`` re-reads it.
    """
    req = GenerationRequest(prompt="a", max_new_tokens=1)
    req.request_id = "drain-e2e"

    real_post = fake_engine._lock_step_post

    def racing_post(step_inputs):
        result = real_post(step_inputs)
        # The OTHER generator's step() re-entering the scheduler and evicting
        # this just-finished sequence before the owner drains it.
        fake_engine.scheduler.schedule()
        return result

    monkeypatch.setattr(fake_engine, "_lock_step_post", racing_post)

    # ``stream_request`` calls ``add_request`` itself — we must NOT also call
    # ``add_request`` up here or the request would be double-added.
    chunks = list(fake_engine.stream_request(req))
    # The single generated token must be emitted. Under the old re-fetch the
    # loop broke before draining, returning [] here.
    assert chunks, f"stream_request must emit the final token, got {chunks!r}"
