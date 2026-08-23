"""Tests for the StreamingLLM attention-sink mask builder (TASK-238 / 15.1)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

from llm.core.attn.block_sparse import coverage_fraction, mask_to_additive
from llm.core.attn.streaming_llm import build_streamingllm_mask


def test_build_validates_arguments():
    with pytest.raises(ValueError, match="num_sink"):
        build_streamingllm_mask(8, num_sink=-1, window_size=2)
    with pytest.raises(ValueError, match="window_size"):
        build_streamingllm_mask(8, num_sink=2, window_size=-1)


def test_sink_covers_all_equals_dense_causal():
    """num_sink >= seq_len -> every position is a sink -> the mask is exactly
    the dense causal mask (streaming is a constraint on dense attention)."""
    seq_len = 16
    mask = build_streamingllm_mask(seq_len, num_sink=seq_len, window_size=0)
    row = torch.arange(seq_len).view(-1, 1)
    col = torch.arange(seq_len).view(1, -1)
    assert torch.equal(mask, (col <= row))


def test_window_covers_all_equals_dense_causal():
    """A window larger than the whole context, with no sink, also reproduces the
    dense causal mask."""
    seq_len = 16
    mask = build_streamingllm_mask(seq_len, num_sink=0, window_size=seq_len)
    row = torch.arange(seq_len).view(-1, 1)
    col = torch.arange(seq_len).view(1, -1)
    assert torch.equal(mask, (col <= row))


def test_sink_positions_always_attended():
    seq_len, num_sink = 16, 2
    mask = build_streamingllm_mask(seq_len, num_sink=num_sink, window_size=4)
    # Every query attends to the sink columns 0..num_sink-1 that are in its past.
    assert bool(mask[15, 0].item())
    assert bool(mask[15, 1].item())
    # A position outside the sink and outside the recent window is blocked.
    assert not bool(mask[10, 5].item())  # token 5 is old (not sink, not in row-10 window)
    # A recent position inside the window is attended.
    assert bool(mask[10, 9].item())


def test_causal_blocks_future_even_when_sink():
    """A position is a sink only for keys in the past; a future key is still
    blocked even if it lies inside the sink range."""
    seq_len, num_sink = 8, 5
    mask = build_streamingllm_mask(seq_len, num_sink=num_sink, window_size=0, causal=True)
    assert not bool(mask[2, 4].item())  # col 4 is a sink but is the future of query 2
    assert bool(mask[4, 2].item())  # sink key 2 in the past of query 4


def test_sparse_old_tokens_contribute_no_attention():
    """Blocked old interior tokens get -inf in the additive mask, so their
    softmax weight is ~0."""
    seq_len, num_sink, window_size = 16, 2, 4
    mask = build_streamingllm_mask(seq_len, num_sink=num_sink, window_size=window_size)
    assert coverage_fraction(mask) < 1.0
    additive = mask_to_additive(mask)
    weights = functional.softmax(torch.zeros(seq_len, seq_len) + additive, dim=-1)
    blocked = ~mask
    assert weights[blocked].abs().max().item() < 1e-6
    assert torch.allclose(weights.sum(-1), torch.ones(seq_len), atol=1e-5)


def test_non_causal_sink_and_symmetric_window():
    seq_len, num_sink, window_size = 16, 2, 2
    mask = build_streamingllm_mask(seq_len, num_sink=num_sink, window_size=window_size, causal=False)
    # Symmetric window attends to one neighbour on each side of the query.
    assert bool(mask[8, 6].item())  # 2 back
    assert not bool(mask[8, 3].item())  # 5 back, not sink -> blocked
    assert bool(mask[10, 1].item())  # sink column 1 is attended by any query
    assert not bool(mask[10, 3].item())  # col 3 is not a sink and off-window -> blocked
