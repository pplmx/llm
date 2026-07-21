"""Tests for Multi-Latent Attention's ``prefix_kv`` extension (T2 PEFT multi-backend).

``MultiLatentAttention.forward`` must accept an optional ``prefix_kv``
argument and prepend it to K and V before ``_latent_attention`` runs. The
full Prefix Tuning wrapper is tested in ``test_prefix_tuning.py``.

MLA-specific concerns covered here (beyond the MHA contract):

1. ``attn_mask`` S_k axis grows by ``prefix_len`` ones — the prefix is
   unconditionally visible regardless of the input mask.
2. Paged cache + prefix runs through the same concat-after-gather path
   as the linear cache; the prefix never enters the paged cache.
3. MLA has no GQA (``num_heads == num_kv_heads``), so the GQA-repeat
   test from MHA is omitted.
"""

from __future__ import annotations

import pytest
import torch

from llm.core.attn.mla import MultiLatentAttention


class TestMLAPrefixKV:
    """``MultiLatentAttention.forward`` must accept ``prefix_kv=None`` with no behavior change."""

    def test_forward_without_prefix_kv_matches_original(self):
        """Baseline: no prefix → forward shape matches input."""
        torch.manual_seed(0)
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8)
        x = torch.randn(2, 5, 32)
        out = attn(x)
        assert out.shape == (2, 5, 32)

    def test_forward_with_zero_prefix_shrinks_output_magnitude(self):
        """Zero-valued V prefix → output magnitude shrinks (prefix rows participate in latent softmax).

        The latent queries attend to (prefix + dynamic) K, V. With V=0,
        the prefix rows still consume softmax mass from the dynamic
        tokens, so the latent output magnitude shrinks. Verifies the
        prefix rows are actually being concatenated into the latent
        attention.
        """
        torch.manual_seed(0)
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8)
        x = torch.randn(2, 5, 32)

        out_no_prefix = attn(x).detach().clone()

        prefix_len = 3
        head_dim = 8  # 32 / 4
        zero_pk = torch.zeros(2, 4, prefix_len, head_dim)
        zero_pv = torch.zeros(2, 4, prefix_len, head_dim)
        out_with_zero_prefix = attn(x, prefix_kv=(zero_pk, zero_pv))

        mag_no = out_no_prefix.abs().mean()
        mag_zero = out_with_zero_prefix.abs().mean()
        ratio = (mag_zero / mag_no).item()
        assert 0.0 < ratio < 1.0, f"expected ratio in (0, 1), got {ratio}"
        assert ratio < 0.99, f"zero prefix should shrink output by >=1%, got {ratio}"
        assert not torch.allclose(out_no_prefix, out_with_zero_prefix, atol=1e-4)

    def test_forward_with_prefix_kv_changes_output(self):
        """Non-zero prefix → output diverges from the no-prefix path."""
        torch.manual_seed(0)
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8)
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
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8)
        x = torch.randn(2, 5, 32)
        head_dim = 8
        for p_len in (1, 3, 8):
            pk = torch.randn(2, 4, p_len, head_dim)
            pv = torch.randn(2, 4, p_len, head_dim)
            out = attn(x, prefix_kv=(pk, pv))
            assert out.shape == (2, 5, 32)

    def test_prefix_kv_mismatched_shape_raises(self):
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8)
        x = torch.randn(2, 5, 32)
        pk = torch.randn(2, 4, 3, 8)
        pv = torch.randn(2, 4, 5, 8)  # different seq dim
        with pytest.raises(ValueError, match="share shape"):
            attn(x, prefix_kv=(pk, pv))

    def test_prefix_kv_wrong_num_kv_heads_raises(self):
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8)
        x = torch.randn(2, 5, 32)
        pk = torch.randn(2, 3, 3, 8)  # 3 != num_kv_heads=4
        pv = torch.randn(2, 3, 3, 8)
        with pytest.raises(ValueError, match="num_kv_heads"):
            attn(x, prefix_kv=(pk, pv))

    def test_prefix_kv_wrong_head_dim_raises(self):
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8)
        x = torch.randn(2, 5, 32)
        pk = torch.randn(2, 4, 3, 16)  # 16 != head_dim=8
        pv = torch.randn(2, 4, 3, 16)
        with pytest.raises(ValueError, match="head_dim"):
            attn(x, prefix_kv=(pk, pv))

    def test_prefix_kv_wrong_batch_raises(self):
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8)
        x = torch.randn(2, 5, 32)
        pk = torch.randn(3, 4, 3, 8)  # batch=3 != 2
        pv = torch.randn(3, 4, 3, 8)
        with pytest.raises(ValueError, match="batch"):
            attn(x, prefix_kv=(pk, pv))


class TestMLAPrefixWithMask:
    """MLA-specific: ``attn_mask`` S_k axis grows by ``prefix_len`` ones (prefix always visible)."""

    def test_mask_axis_grows_with_prefix(self):
        """Forward succeeds when prefix is prepended and a mask is provided.

        Without mask widening the latent attention's mask reshape at
        ``_latent_attention`` would mismatch the extended K shape and
        blow up with a shape error.
        """
        torch.manual_seed(0)
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8)
        x = torch.randn(2, 5, 32)

        # Mask covers the dynamic tokens: shape [B, 1, S_q, S_k] = [2, 1, 1, 5]
        # (S_q=1 because MLA collapses to the last query position).
        attn_mask = torch.ones(2, 1, 1, 5)

        prefix_len = 3
        head_dim = 8
        pk = torch.randn(2, 4, prefix_len, head_dim)
        pv = torch.randn(2, 4, prefix_len, head_dim)

        out = attn(x, attn_mask=attn_mask, prefix_kv=(pk, pv))
        assert out.shape == (2, 5, 32)

    def test_prefix_always_visible_regardless_of_mask(self):
        """A zero input mask still allows prefix-attended output (prefix segment is 1s).

        We construct a mask that zeros out the dynamic tokens, then
        verify the output with prefix is non-zero — the prefix segment
        was widened with ones regardless of the input mask.
        """
        torch.manual_seed(0)
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8)
        x = torch.randn(2, 5, 32)

        # Mask that hides the dynamic tokens entirely.
        attn_mask = torch.zeros(2, 1, 1, 5)

        prefix_len = 3
        head_dim = 8
        pk = torch.randn(2, 4, prefix_len, head_dim)
        pv = torch.randn(2, 4, prefix_len, head_dim)

        out = attn(x, attn_mask=attn_mask, prefix_kv=(pk, pv))
        # Output should be non-zero because the prefix segment
        # contributes via the ones we prepended to the mask.
        assert out.abs().mean() > 1e-6


class TestMLAPrefixWithLinearCache:
    """MLA linear KVCache + prefix interaction."""

    def test_prefix_with_linear_kv_cache_runs(self):
        """Forward succeeds with both kv_cache and prefix_kv set."""
        from llm.core.kv_cache import KVCache

        torch.manual_seed(0)
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8)
        x = torch.randn(1, 5, 32)

        kv_cache = KVCache(
            max_batch_size=1,
            max_seq_len=8,
            num_kv_heads=4,
            head_dim=8,
        )

        prefix_len = 2
        pk = torch.randn(1, 4, prefix_len, 8)
        pv = torch.randn(1, 4, prefix_len, 8)

        out = attn(x, kv_cache=kv_cache, prefix_kv=(pk, pv))
        assert out.shape == (1, 5, 32)
        assert out.isfinite().all()

    def test_cached_kv_does_not_contain_prefix(self):
        """The KV cache stores dynamic tokens only, not the prefix.

        After forward with prefix_kv, the cache's seq_len should equal
        the input's seq_len (5), not seq_len + prefix_len (7).
        """
        from llm.core.kv_cache import KVCache

        torch.manual_seed(0)
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8)
        x = torch.randn(1, 5, 32)

        kv_cache = KVCache(
            max_batch_size=1,
            max_seq_len=8,
            num_kv_heads=4,
            head_dim=8,
        )

        prefix_len = 2
        pk = torch.randn(1, 4, prefix_len, 8)
        pv = torch.randn(1, 4, prefix_len, 8)

        attn(x, kv_cache=kv_cache, prefix_kv=(pk, pv))

        # Cache should hold only the 5 dynamic tokens, not 7.
        assert kv_cache.seq_len == 5


class TestMLAPrefixWithPagedCache:
    """MLA paged KVCache + prefix interaction.

    The prefix is concatenated AFTER the paged gather returns
    (``_paged_kv_write`` returns raw K, V which are then concatenated
    with the prefix). The paged cache itself stores only dynamic tokens.
    """

    def test_prefix_with_paged_cache_runs(self):
        from llm.core.paged_attention.paged_kv_cache import PagedKVCache

        torch.manual_seed(0)
        attn = MultiLatentAttention(hidden_size=32, num_heads=4, num_latents=8)
        x = torch.randn(1, 5, 32)

        paged = PagedKVCache(
            num_layers=1,
            num_kv_heads=4,
            head_dim=8,
            num_blocks=4,
            block_size=4,
            device="cpu",
            dtype=torch.float32,
        )

        prefix_len = 2
        pk = torch.randn(1, 4, prefix_len, 8)
        pv = torch.randn(1, 4, prefix_len, 8)

        out = attn(
            x,
            paged_kv_cache=paged,
            layer_idx=0,
            batch_indices=torch.tensor([0], dtype=torch.long),
            prefix_kv=(pk, pv),
        )
        assert out.shape == (1, 5, 32)
        assert out.isfinite().all()
