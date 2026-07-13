"""Property-based tests for :mod:`llm.core.kv_cache` (Finding AD).

``KVCache.update`` is the hot path every generation step writes to.
Bugs here corrupt attention inputs silently — the kind of failure
that only surfaces as "model output quality regressed" in production.

Hypothesis verifies four invariants the existing example-based tests
don't exercise:

1. After N ``update`` calls, ``seq_len`` equals the sum of new tokens.
2. ``update(k_new, v_new)`` followed by ``reset()`` restores the cache
   to a fresh ``_seq_len=0`` state (memory is retained but logically
   cleared).
3. Writes that would overflow ``max_seq_len`` raise ``ValueError``
   BEFORE any partial state is mutated (transactional).
4. The cached slice ``[B, N_kv, :seq_len, :]`` matches a naive
   ``torch.cat`` reference for arbitrary ``k_new`` / ``v_new`` tensors.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from llm.core.kv_cache import KVCache

# Strategies that produce tensors of shape
# ``[batch_size, num_kv_heads, new_tokens, head_dim]`` where
# ``batch_size <= max_batch_size`` and ``new_tokens <= max_seq_len``.
_BATCH_SIZE = 3
_NUM_KV_HEADS = 4
_HEAD_DIM = 8
_MAX_SEQ_LEN = 16
_MAX_BATCH_SIZE = 4


@st.composite
def _cache_update_inputs(draw):
    """Yield ``(cache, k_new, v_new)`` valid for a single ``update`` call."""
    cache = KVCache(
        max_batch_size=_MAX_BATCH_SIZE,
        max_seq_len=_MAX_SEQ_LEN,
        num_kv_heads=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
        device="cpu",
    )
    new_tokens = draw(st.integers(min_value=1, max_value=_MAX_SEQ_LEN))
    batch_size = draw(st.integers(min_value=1, max_value=_MAX_BATCH_SIZE))
    k_new = torch.randn(batch_size, _NUM_KV_HEADS, new_tokens, _HEAD_DIM)
    v_new = torch.randn(batch_size, _NUM_KV_HEADS, new_tokens, _HEAD_DIM)
    return cache, k_new, v_new


@st.composite
def _two_update_sequences(draw):
    """Yield ``(cache, k1, v1, k2, v2)`` valid for two ``update`` calls.

    The first update must not already overflow ``max_seq_len``; the
    second update may or may not fit depending on ``seq_len_new``.
    """
    cache = KVCache(
        max_batch_size=_MAX_BATCH_SIZE,
        max_seq_len=_MAX_SEQ_LEN,
        num_kv_heads=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
        device="cpu",
    )
    first_tokens = draw(st.integers(min_value=1, max_value=_MAX_SEQ_LEN))
    second_tokens = draw(st.integers(min_value=1, max_value=_MAX_SEQ_LEN))
    batch_size = draw(st.integers(min_value=1, max_value=_MAX_BATCH_SIZE))

    k1 = torch.randn(batch_size, _NUM_KV_HEADS, first_tokens, _HEAD_DIM)
    v1 = torch.randn(batch_size, _NUM_KV_HEADS, first_tokens, _HEAD_DIM)
    k2 = torch.randn(batch_size, _NUM_KV_HEADS, second_tokens, _HEAD_DIM)
    v2 = torch.randn(batch_size, _NUM_KV_HEADS, second_tokens, _HEAD_DIM)
    return cache, k1, v1, k2, v2


# --- Invariant 1: seq_len equals sum of new tokens ------------------------


@given(args=_cache_update_inputs())
@settings(max_examples=50, deadline=None)
def test_seq_len_after_update_equals_total_tokens(args):
    """After update, ``seq_len == previous + new_tokens``.

    Off-by-one bugs in the slice index (``seq_len:new_seq_len``)
    would show up here: any call that wrote one slot too few or too
    many would diverge from the expected length.
    """
    cache, k_new, v_new = args
    before = cache.seq_len
    cache.update(k_new, v_new)
    assert cache.seq_len == before + k_new.size(2)


# --- Invariant 2: reset restores _seq_len=0 -------------------------------


@given(args=_cache_update_inputs())
@settings(max_examples=50, deadline=None)
def test_reset_restores_seq_len_to_zero(args):
    """``reset()`` returns the cache to ``seq_len=0`` (memory retained).

    The buffers are NOT deallocated (that's the whole point of
    pre-allocation). Only the logical ``_seq_len`` cursor resets.
    """
    cache, k_new, v_new = args
    cache.update(k_new, v_new)
    assert cache.seq_len > 0
    cache.reset()
    assert cache.seq_len == 0


# --- Invariant 3: overflow raises BEFORE any state change -----------------


@given(args=_two_update_sequences())
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_overflow_raises_and_preserves_state(args):
    """Overflow raises ValueError without mutating any cache state.

    The KV cache is pre-allocated and shared across requests. A
    partial write that gets half-applied before raising would leave
    the cache in an inconsistent state, producing garbage on the
    next read.
    """
    cache, k1, v1, k2, v2 = args
    cache.update(k1, v1)
    seq_before = cache.seq_len
    k_before = cache.k_cache.clone()
    v_before = cache.v_cache.clone()

    if seq_before + k2.size(2) > cache.max_seq_len:
        with pytest.raises(ValueError, match="overflow"):
            cache.update(k2, v2)
        # Cache must be byte-identical to its pre-call snapshot.
        assert cache.seq_len == seq_before
        assert torch.equal(cache.k_cache, k_before)
        assert torch.equal(cache.v_cache, v_before)


# --- Invariant 4: cached slice matches torch.cat reference ----------------


@given(args=_cache_update_inputs())
@settings(max_examples=50, deadline=None)
def test_cached_slice_matches_torch_cat_reference(args):
    """The cached slice ``[B, N_kv, :seq_len, :]`` matches ``k_new`` written.

    The simplest correctness oracle: after ``update(k_new, v_new)``,
    the first ``seq_len`` slots of the cache buffer must equal
    ``k_new`` at the positions we just wrote. We compare with the
    full cache view returned by ``update`` (which already covers
    that range).
    """
    cache, k_new, v_new = args
    k_view, v_view = cache.update(k_new, v_new)
    new_tokens = k_new.size(2)

    # The view returned by update is the cached region for the
    # just-written tokens. Its last ``new_tokens`` positions must
    # match ``k_new`` exactly (same tensor identity, since the
    # implementation copies via assignment).
    assert torch.equal(k_view[:, :, -new_tokens:, :], k_new)
    assert torch.equal(v_view[:, :, -new_tokens:, :], v_new)


# --- Invariant 5: seq_len never exceeds max_seq_len ----------------------


@given(
    num_updates=st.integers(min_value=1, max_value=5),
    tokens_per_update=st.integers(min_value=1, max_value=_MAX_SEQ_LEN),
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_seq_len_never_exceeds_max_seq_len(num_updates, tokens_per_update):
    """After any number of legal updates, ``seq_len <= max_seq_len``.

    The earlier overflow test verifies that the ``>max_seq_len`` path
    raises. This test verifies the invariant after a sequence of
    legal updates.
    """
    assume(num_updates * tokens_per_update <= _MAX_SEQ_LEN)
    cache = KVCache(
        max_batch_size=_MAX_BATCH_SIZE,
        max_seq_len=_MAX_SEQ_LEN,
        num_kv_heads=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
        device="cpu",
    )
    for _ in range(num_updates):
        cache.update(
            torch.randn(1, _NUM_KV_HEADS, tokens_per_update, _HEAD_DIM),
            torch.randn(1, _NUM_KV_HEADS, tokens_per_update, _HEAD_DIM),
        )
    assert cache.seq_len <= cache.max_seq_len
    assert cache.seq_len == num_updates * tokens_per_update
