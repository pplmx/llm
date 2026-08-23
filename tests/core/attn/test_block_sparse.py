"""Tests for Block Sparse Attention mask builder (TASK-237 / ROADMAP 15.2)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

from llm.core.attn.block_sparse import (
    build_block_sparse_mask,
    coverage_fraction,
    mask_to_additive,
)


def test_build_validates_arguments():
    with pytest.raises(ValueError, match="block_size"):
        build_block_sparse_mask(8, block_size=0, window_blocks=1, global_blocks=1)
    with pytest.raises(ValueError, match="random_blocks"):
        build_block_sparse_mask(8, block_size=2, window_blocks=1, global_blocks=1, random_blocks=99)


def test_full_global_coverage_equals_dense_causal():
    """When global blocks cover every key block, the block-sparse mask is
    exactly the full causal mask (block sparsity is a constraint on top of
    dense attention, not a different computation)."""
    seq_len, block_size = 8, 2  # 4 blocks
    mask = build_block_sparse_mask(seq_len, block_size=block_size, window_blocks=0, global_blocks=4)
    row = torch.arange(seq_len).view(-1, 1)
    col = torch.arange(seq_len).view(1, -1)
    assert torch.equal(mask, (col <= row))
    assert coverage_fraction(mask) == pytest.approx((seq_len * (seq_len + 1) / 2) / (seq_len * seq_len))


def test_full_window_coverage_equals_dense_causal():
    """Window covering every block also reproduces the dense causal mask."""
    seq_len, block_size = 8, 2  # 4 blocks
    mask = build_block_sparse_mask(seq_len, block_size=block_size, window_blocks=3, global_blocks=0)
    row = torch.arange(seq_len).view(-1, 1)
    col = torch.arange(seq_len).view(1, -1)
    assert torch.equal(mask, (col <= row))


def test_full_coverage_non_causal_is_all_ones():
    seq_len, block_size = 8, 2
    mask = build_block_sparse_mask(seq_len, block_size=block_size, window_blocks=0, global_blocks=4, causal=False)
    assert bool(mask.all())


def test_global_blocks_are_always_attended():
    """Position 0 is a global token: every query attends to it (both causal
    and non-causal), and it attends to every key."""
    seq_len, block_size = 8, 2
    mask = build_block_sparse_mask(seq_len, block_size=block_size, window_blocks=0, global_blocks=1, causal=False)
    # Column 0 and row 0 all True (global token attended everywhere / attends all).
    assert bool(mask[:, 0].all())
    assert bool(mask[0, :].all())
    # A non-global, off-window position is blocked: query block 3 vs key block 1.
    assert not bool(mask[7, 3])


def test_genuinely_sparse_blocks_contribute_no_attention():
    """Blocked positions get -inf in the additive mask, so softmax weight ~ 0."""
    seq_len, block_size = 8, 2
    mask = build_block_sparse_mask(seq_len, block_size=block_size, window_blocks=1, global_blocks=1)
    additive = mask_to_additive(mask)
    scores = torch.zeros(seq_len, seq_len)
    weights = functional.softmax(scores + additive, dim=-1)
    blocked = ~mask
    assert weights[blocked].abs().max().item() < 1e-6
    # And the attended positions still normalize to a probability distribution.
    assert torch.allclose(weights.sum(-1), torch.ones(seq_len), atol=1e-5)


def test_causal_blocks_future_keys():
    seq_len, block_size = 8, 2
    mask = build_block_sparse_mask(seq_len, block_size=block_size, window_blocks=2, global_blocks=1, causal=True)
    assert not bool(mask[0, 1])  # position 1 is in the future of 0
    assert bool(mask[1, 1])
    assert bool(mask[5, 3])  # local past position within the window


def test_random_blocks_are_deterministic():
    seq_len, block_size = 8, 2  # 4 blocks
    a = build_block_sparse_mask(
        seq_len, block_size=block_size, window_blocks=0, global_blocks=0, random_blocks=2, seed=3
    )
    b = build_block_sparse_mask(
        seq_len, block_size=block_size, window_blocks=0, global_blocks=0, random_blocks=2, seed=3
    )
    assert torch.equal(a, b)
    c = build_block_sparse_mask(
        seq_len, block_size=block_size, window_blocks=0, global_blocks=0, random_blocks=2, seed=4
    )
    assert not torch.equal(a, c)


def test_mask_to_additive_validation():
    with pytest.raises(ValueError, match="bool"):
        mask_to_additive(torch.zeros(2, 2))
