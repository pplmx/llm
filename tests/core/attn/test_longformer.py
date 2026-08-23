"""Tests for the Longformer-style dilated sliding-window mask (TASK-240 / 15.2)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

from llm.core.attn import MultiHeadAttention
from llm.core.attn.block_sparse import coverage_fraction, mask_to_additive
from llm.core.attn.longformer import build_longformer_mask
from llm.core.attn.sdpa import sdpa

SEQ, B, N, D = 16, 2, 4, 16


def test_build_validates_arguments():
    with pytest.raises(ValueError, match="window_size"):
        build_longformer_mask(8, window_size=-1)
    with pytest.raises(ValueError, match="dilation"):
        build_longformer_mask(8, window_size=4, dilation=0)


def test_dilation_one_equals_plain_window():
    """dilation=1 + no global reproduces the plain causal sliding window exactly."""
    seq_len, window_size = 16, 6
    mask = build_longformer_mask(seq_len, window_size=window_size, dilation=1, num_global=0, causal=True)
    row = torch.arange(seq_len).view(-1, 1)
    col = torch.arange(seq_len).view(1, -1)
    expected = (col <= row) & (col >= row - window_size)
    assert torch.equal(mask, expected)


def test_global_tokens_always_attended():
    seq_len, num_global = 16, 2
    mask = build_longformer_mask(seq_len, window_size=4, dilation=2, num_global=num_global, causal=True)
    assert bool(mask[15, 0].item())  # global key 0 attended by the last query
    assert bool(mask[15, 1].item())  # global key 1 attended by the last query


def test_dilation_reduces_coverage():
    """Within the same window, dilation>1 samples fewer positions."""
    seq_len, window_size = 32, 12
    dense = build_longformer_mask(seq_len, window_size=window_size, dilation=1, num_global=0, causal=True)
    dilated = build_longformer_mask(seq_len, window_size=window_size, dilation=3, num_global=0, causal=True)
    assert coverage_fraction(dilated) < coverage_fraction(dense)
    # The dilated mask is a subset of the dense window mask: nothing is
    # attending to a key the dense window does not.
    assert bool((~dilated | dense).all())


def test_causal_blocks_future():
    seq_len, window_size = 16, 6
    mask = build_longformer_mask(seq_len, window_size=window_size, dilation=1, num_global=8, causal=True)
    assert not bool(mask[2, 9].item())  # future key 9 even though it is a global position


def test_sdpa_integration_matches_explicit_masking():
    torch.manual_seed(0)
    q = torch.randn(B, N, SEQ, D)
    k = torch.randn(B, N, SEQ, D)
    v = torch.randn(B, N, SEQ, D)
    mask = build_longformer_mask(SEQ, window_size=6, dilation=2, num_global=2, causal=True)
    mha_mask = (~mask).bool().unsqueeze(0).unsqueeze(0)  # True = mask out
    out = sdpa(q, k, v, attn_mask=mha_mask, is_causal=False)
    additive = mask_to_additive(mask).to(q.dtype).unsqueeze(0).unsqueeze(0)
    ref = functional.scaled_dot_product_attention(q, k, v, attn_mask=additive, is_causal=False)
    assert torch.allclose(out, ref, atol=1e-5)


def test_mha_forward_sparse_differs_and_full_coverage_parity():
    mha = MultiHeadAttention(hidden_size=64, num_heads=8, is_causal=False, include_norm_residual=True, p=0.0)
    hidden = torch.randn(B, SEQ, 64)
    dense = mha(hidden, attn_mask=None, is_causal=False)
    dense = dense[0] if isinstance(dense, tuple) else dense

    sparse = build_longformer_mask(SEQ, window_size=4, dilation=2, num_global=1, causal=False)
    out = mha(hidden, attn_mask=(~sparse).bool().unsqueeze(0).unsqueeze(0), is_causal=False)
    out = out[0] if isinstance(out, tuple) else out
    assert not torch.allclose(out, dense, atol=1e-4), "dilated mask changed nothing in the real forward"

    full = build_longformer_mask(SEQ, window_size=SEQ, dilation=1, num_global=SEQ, causal=False)
    out_full = mha(hidden, attn_mask=(~full).bool().unsqueeze(0).unsqueeze(0), is_causal=False)
    out_full = out_full[0] if isinstance(out_full, tuple) else out_full
    assert torch.allclose(dense, out_full, atol=1e-6)
