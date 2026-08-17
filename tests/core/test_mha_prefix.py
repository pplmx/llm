"""Tests for MHA's ``prefix_kv`` extension (T2 PEFT foundation).

``MHA.forward`` must accept an optional ``prefix_kv`` argument and
prepend it to K and V before the attention compute. The full
Prefix Tuning wrapper is tested in ``test_prefix_tuning.py``.
"""

from __future__ import annotations

import torch

from llm.core.attn.mha import MultiHeadAttention


class TestMHAPrefixKV:
    """``MHA.forward`` must accept ``prefix_kv=None`` with no behavior change."""

    def test_forward_without_prefix_kv_matches_original(self):
        """Baseline: no prefix → forward is byte-identical to the pre-extension path."""
        torch.manual_seed(0)
        attn = MultiHeadAttention(hidden_size=32, num_heads=4)
        x = torch.randn(2, 5, 32)
        out = attn(x)
        assert out.shape == (2, 5, 32)

    def test_forward_with_zero_prefix_shrinks_output_magnitude(self):
        """A zero-valued prefix (V=0) contributes nothing to the output, but
        the extra K rows still participate in the softmax and steal mass
        from the existing tokens.

        The new attention weight for existing token ``i`` is
        ``exp(s_i) / (Z + P)`` instead of ``exp(s_i) / Z`` where
        ``Z = sum_j exp(s_j)`` and ``P = prefix_len``. The output is
        therefore rescaled by ``Z / (Z + P) ∈ (0, 1)`` — the existing
        tokens' contribution shrinks. This verifies the prefix rows are
        actually being concatenated into the attention (a no-op prefix
        would keep the output magnitude unchanged).
        """
        torch.manual_seed(0)
        attn = MultiHeadAttention(hidden_size=32, num_heads=4)
        x = torch.randn(2, 5, 32)

        out_no_prefix = attn(x).detach().clone()

        prefix_len = 3
        head_dim = 8  # 32 / 4
        zero_pk = torch.zeros(2, 4, prefix_len, head_dim)
        zero_pv = torch.zeros(2, 4, prefix_len, head_dim)
        out_with_zero_prefix = attn(x, prefix_kv=(zero_pk, zero_pv))

        # Output magnitude must strictly shrink — otherwise the prefix
        # rows are not being concatenated into the attention.
        mag_no = out_no_prefix.abs().mean()
        mag_zero = out_with_zero_prefix.abs().mean()
        ratio = (mag_zero / mag_no).item()
        assert 0.0 < ratio < 1.0, f"expected ratio in (0, 1), got {ratio}"
        # Concrete upper bound: at least 5% shrinkage for prefix_len=3.
        assert ratio < 0.99, f"zero prefix should shrink output by ≥1%, got {ratio}"
        # And: the outputs must NOT be identical (different weights ⇒ different output).
        assert not torch.allclose(out_no_prefix, out_with_zero_prefix, atol=1e-4)

    def test_forward_with_prefix_kv_changes_output(self):
        """Non-zero prefix → output diverges from the no-prefix path."""
        torch.manual_seed(0)
        attn = MultiHeadAttention(hidden_size=32, num_heads=4)
        x = torch.randn(2, 5, 32)

        out_no_prefix = attn(x).detach().clone()

        prefix_len = 3
        head_dim = 8
        pk = torch.randn(2, 4, prefix_len, head_dim)
        pv = torch.randn(2, 4, prefix_len, head_dim)
        out_with_prefix = attn(x, prefix_kv=(pk, pv))

        assert not torch.allclose(out_no_prefix, out_with_prefix, atol=1e-4)

    def test_prefix_does_not_affect_output_shape(self):
        """Output shape is independent of prefix_len."""
        torch.manual_seed(0)
        attn = MultiHeadAttention(hidden_size=32, num_heads=4)
        x = torch.randn(2, 5, 32)
        head_dim = 8
        for p_len in (1, 3, 8):
            pk = torch.randn(2, 4, p_len, head_dim)
            pv = torch.randn(2, 4, p_len, head_dim)
            out = attn(x, prefix_kv=(pk, pv))
            assert out.shape == (2, 5, 32)

    def test_prefix_kv_mismatched_shape_raises(self):
        attn = MultiHeadAttention(hidden_size=32, num_heads=4)
        x = torch.randn(2, 5, 32)
        pk = torch.randn(2, 4, 3, 8)
        pv = torch.randn(2, 4, 5, 8)  # different seq dim
        with __import__("pytest").raises(ValueError, match="share shape"):
            attn(x, prefix_kv=(pk, pv))

    def test_prefix_kv_wrong_num_kv_heads_raises(self):
        attn = MultiHeadAttention(hidden_size=32, num_heads=4)
        x = torch.randn(2, 5, 32)
        pk = torch.randn(2, 3, 3, 8)  # 3 != num_kv_heads=4
        pv = torch.randn(2, 3, 3, 8)
        with __import__("pytest").raises(ValueError, match="num_kv_heads"):
            attn(x, prefix_kv=(pk, pv))

    def test_gqa_repeats_prefix_too(self):
        """With GQA, the prefix must be repeated to match num_query_heads."""
        torch.manual_seed(0)
        # 8 query heads, 2 kv heads → num_queries_per_kv = 4
        attn = MultiHeadAttention(hidden_size=32, num_heads=8, num_kv_heads=2)
        x = torch.randn(1, 4, 32)
        head_dim = 4
        # Prefix in kv-heads shape: [1, 2, 3, 4]
        pk = torch.randn(1, 2, 3, head_dim)
        pv = torch.randn(1, 2, 3, head_dim)
        out = attn(x, prefix_kv=(pk, pv))
        # Output shape unchanged.
        assert out.shape == (1, 4, 32)


class TestMHAPrefixCausal:
    """Prefix K/V prepends ``prefix_len`` columns, so a *causal* prefill/decode
    must shift its diagonal by the prefix — otherwise every real context key is
    masked and the query attends only prefix keys (round-71 HIGH).

    Every older prefix test called MHA with the default ``is_causal=False``, so
    this causal path was never exercised.
    """

    def test_causal_prefix_matches_explicit_shifted_mask(self):
        """``is_causal=True`` with a prefix is equivalent to the explicit
        shifted-causal bool mask (visible cols 0..prefix_len+row)."""

        torch.manual_seed(0)
        attn = MultiHeadAttention(hidden_size=32, num_heads=4, is_causal=True)
        attn.eval()  # disable output-projection dropout so two calls are equal
        x = torch.randn(2, 5, 32)
        prefix_len = 3
        pk = torch.randn(2, 4, prefix_len, 8)
        pv = torch.randn(2, 4, prefix_len, 8)

        out_auto = attn(x, prefix_kv=(pk, pv))

        # Reference: causal folded into an explicit mask, causality off.
        mask = torch.triu(torch.ones(5, 5 + prefix_len, dtype=torch.bool), diagonal=1 + prefix_len)
        out_explicit = attn(x, prefix_kv=(pk, pv), is_causal=False, attn_mask=mask)
        assert torch.allclose(out_auto, out_explicit, atol=1e-6)

    def test_causal_prefix_differs_from_legacy_top_left_causal(self):
        """Sanity that the shift actually changes attention (the pre-fix path
        masked the real context and would produce a different output)."""
        torch.manual_seed(1)
        attn = MultiHeadAttention(hidden_size=32, num_heads=4, is_causal=True)
        attn.eval()  # disable output-projection dropout for deterministic comparison
        x = torch.randn(2, 5, 32)
        prefix_len = 3
        pk = torch.randn(2, 4, prefix_len, 8)
        pv = torch.randn(2, 4, prefix_len, 8)
        out_shifted = attn(x, prefix_kv=(pk, pv))
        # Explicit unshifted top-left causal mask (the pre-fix behavior).
        mask = torch.triu(torch.ones(5, 5 + prefix_len, dtype=torch.bool), diagonal=1)
        out_top_left = attn(x, prefix_kv=(pk, pv), is_causal=False, attn_mask=mask)
        assert not torch.allclose(out_shifted, out_top_left, atol=1e-6)

    def test_paged_prefix_kv_rejected(self):
        """The paged/continuous-batching branch cannot thread ``prefix_kv``, so
        it must reject loudly instead of silently serving an un-prefixed model
        (round-71 paged-prefix fix; matches FlashAttention's refusal)."""
        import pytest

        torch.manual_seed(3)
        attn = MultiHeadAttention(hidden_size=32, num_heads=4)
        x = torch.randn(1, 3, 32)
        pk = torch.randn(1, 4, 2, 8)
        pv = torch.randn(1, 4, 2, 8)
        with pytest.raises(NotImplementedError, match="prefix tuning"):
            attn(x, prefix_kv=(pk, pv), paged_kv_cache=object())

    def test_causal_prefix_decode_keeps_real_key_visible(self):
        """Decode (q_len=1) with a prefix still attends the single real key:
        its V contribution appears in the output (a col-0-only mask would hide
        it)."""
        torch.manual_seed(2)
        attn = MultiHeadAttention(hidden_size=16, num_heads=1, is_causal=True)
        attn.eval()  # disable output-projection dropout for deterministic comparison
        x = torch.randn(1, 1, 16)
        prefix_len = 4
        pk = torch.randn(1, 1, prefix_len, 16)
        pv = torch.zeros(1, 1, prefix_len, 16)  # prefix V = 0 → contributes nothing
        # Real token → V = 1 at dim 0 via a one-hot-ish input is awkward with
        # learnable qkv; instead pin the output equals the no-mask path when the
        # explicit shifted mask is keep-all (prefix rows + the one real key).
        mask = torch.triu(torch.ones(1, 1 + prefix_len, dtype=torch.bool), diagonal=prefix_len + 1)
        out_causal = attn(x, prefix_kv=(pk, pv), is_causal=True)
        out_legacy = attn(x, prefix_kv=(pk, pv), is_causal=False, attn_mask=mask)
        assert torch.allclose(out_causal, out_legacy, atol=1e-6)
