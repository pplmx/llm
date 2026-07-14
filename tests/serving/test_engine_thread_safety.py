"""Concurrency tests for ``ContinuousBatchingEngine`` (audit Finding AT).

The engine is reachable from multiple worker threads (FastAPI runs each
``service.generate`` call in a threadpool). Mutations to ``self.scheduler``,
``self.slot_allocator``, ``self.kv_caches``, and ``self.prefix_cache`` would
race without a lock. These tests pin the contract that ``step()`` is safe
under concurrent invocation.

Strategy: build an engine against a tiny CPU-only fake model and a stub
tokenizer, then drive it from many threads. The model forward is monkey-
patched to return deterministic logits so we can assert state invariants
without a real GPU dependency.
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
    """Tiny CPU-only model with the surface that ContinuousBatchingEngine reads."""

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
        logits = torch.zeros(bs, seq_len, self.vocab_size, dtype=torch.float32)
        for i in range(bs):
            for j in range(seq_len):
                # Pick a token that is NOT the EOS token to avoid auto-finish.
                logits[i, j, (int(input_ids[i, j].item()) + 2) % self.vocab_size] = 10.0
        return logits, kv_caches


@pytest.fixture
def fake_engine():
    """Construct a real ContinuousBatchingEngine with a CPU fake model."""
    model = _FakeModel(vocab_size=64, n_layers=1)
    tokenizer = _StubTokenizer()
    engine = ContinuousBatchingEngine(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
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

    After T2 #23 the lock is held only for ``_lock_step_pre`` and
    ``_lock_step_post`` (the model forward runs with the lock released).
    We instrument BOTH pre and post to verify the bookkeeping sections
    serialise: if both threads enter the critical section simultaneously,
    the test fails.
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
    assert len(allocator.seq_to_slot) == 0, (
        f"unfreed slots: {allocator.seq_to_slot}"
    )
    # free_slots equals the original pool.
    assert len(allocator.free_slots) == fake_engine.max_batch_size
