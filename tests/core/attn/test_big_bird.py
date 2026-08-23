"""Tests for the BigBird attention mask builder (TASK-241 / ROADMAP 15.2)."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from llm.core.attn import MultiHeadAttention
from llm.core.attn.big_bird import build_bigbird_mask, mask_to_additive
from llm.core.attn.block_sparse import build_block_sparse_mask, coverage_fraction
from llm.core.attn.sdpa import sdpa

SEQ, B, N, D = 24, 2, 4, 16


def test_delegates_to_block_sparse():
    """The BigBird API is literally the block-sparse global+window+random config."""
    a = build_bigbird_mask(SEQ, block_size=4, num_global_blocks=2, window_blocks=1, num_random_blocks=2, seed=3)
    b = build_block_sparse_mask(SEQ, block_size=4, window_blocks=1, global_blocks=2, random_blocks=2, seed=3)
    assert torch.equal(a, b)


def test_full_global_coverage_equals_dense():
    seq_len, block_size = 24, 4  # 6 blocks
    mask = build_bigbird_mask(seq_len, block_size=block_size, num_global_blocks=6, window_blocks=2)
    row = torch.arange(seq_len).view(-1, 1)
    col = torch.arange(seq_len).view(1, -1)
    assert torch.equal(mask, (col <= row))
    full_nc = build_bigbird_mask(seq_len, block_size=block_size, num_global_blocks=6, window_blocks=0, causal=False)
    assert bool(full_nc.all())


def test_random_blocks_add_coverage():
    seq_len, block_size = 32, 4  # 8 blocks
    base = build_bigbird_mask(seq_len, block_size=block_size, num_global_blocks=2, window_blocks=1, num_random_blocks=0)
    with_random = build_bigbird_mask(
        seq_len, block_size=block_size, num_global_blocks=2, window_blocks=1, num_random_blocks=2, seed=1
    )
    assert not torch.equal(base, with_random)
    assert coverage_fraction(with_random) > coverage_fraction(base)


def test_random_blocks_are_deterministic():
    seq_len, block_size = 32, 4
    a = build_bigbird_mask(
        seq_len, block_size=block_size, num_global_blocks=2, window_blocks=1, num_random_blocks=3, seed=7
    )
    b = build_bigbird_mask(
        seq_len, block_size=block_size, num_global_blocks=2, window_blocks=1, num_random_blocks=3, seed=7
    )
    assert torch.equal(a, b)


def test_causal_blocks_future():
    seq_len, block_size = 24, 4
    mask = build_bigbird_mask(seq_len, block_size=block_size, num_global_blocks=4, window_blocks=1, causal=True)
    row = torch.arange(seq_len).view(-1, 1)
    col = torch.arange(seq_len).view(1, -1)
    assert not bool((mask & (col > row)).any())


def test_sdpa_integration_matches_explicit_masking():
    torch.manual_seed(0)
    q = torch.randn(B, N, SEQ, D)
    k = torch.randn(B, N, SEQ, D)
    v = torch.randn(B, N, SEQ, D)
    mask = build_bigbird_mask(SEQ, block_size=4, num_global_blocks=2, window_blocks=1, num_random_blocks=2, seed=5)
    out = sdpa(q, k, v, attn_mask=(~mask).bool().unsqueeze(0).unsqueeze(0), is_causal=False)
    additive = mask_to_additive(mask).to(q.dtype).unsqueeze(0).unsqueeze(0)
    ref = functional.scaled_dot_product_attention(q, k, v, attn_mask=additive, is_causal=False)
    assert torch.allclose(out, ref, atol=1e-5)


def test_mha_forward_bigbird_changes_output():
    mha = MultiHeadAttention(hidden_size=64, num_heads=8, is_causal=False, include_norm_residual=True, p=0.0)
    hidden = torch.randn(B, SEQ, 64)
    dense = mha(hidden, attn_mask=None, is_causal=False)
    dense = dense[0] if isinstance(dense, tuple) else dense
    mask = build_bigbird_mask(SEQ, block_size=4, num_global_blocks=2, window_blocks=0, num_random_blocks=1, seed=2)
    out = mha(hidden, attn_mask=(~mask).bool().unsqueeze(0).unsqueeze(0), is_causal=False)
    out = out[0] if isinstance(out, tuple) else out
    # Genuinely sparse BigBird must not equal dense attention.
    assert not torch.allclose(out, dense, atol=1e-4)
