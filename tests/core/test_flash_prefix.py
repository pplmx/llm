"""Tests for Flash Attention 2's ``prefix_kv`` extension (T2 PEFT multi-backend).

``FlashAttention.forward`` must accept an optional ``prefix_kv`` argument and
prepend it to K and V before the attention compute, mirroring the MHA
contract. All tests are gated on the optional ``flash-attn`` dependency.

The full Prefix Tuning wrapper is tested in ``test_prefix_tuning.py``.
"""

from __future__ import annotations

import pytest
import torch

from llm.core.attn.flash_attn import FLASH_ATTN_AVAILABLE, FlashAttention

# All tests in this file require flash-attn. CPU-only CI imports the
# module but cannot construct or forward through FlashAttention.
pytestmark = pytest.mark.skipif(
    not FLASH_ATTN_AVAILABLE,
    reason="flash-attn is optional; install via `llm[perf]`",
)


class TestFlashPrefixKV:
    """``FlashAttention.forward`` must accept ``prefix_kv=None`` with no behavior change."""

    def test_forward_without_prefix_kv_matches_original(self):
        """Baseline: no prefix → forward shape matches input."""
        torch.manual_seed(0)
        attn = FlashAttention(hidden_size=32, num_heads=4, dtype=torch.float16)
        x = torch.randn(2, 5, 32, dtype=torch.float16)
        out = attn(x)
        assert out.shape == (2, 5, 32)

    def test_forward_with_zero_prefix_shrinks_output_magnitude(self):
        """Zero-valued V prefix → output magnitude shrinks (prefix rows participate in softmax).

        With V=0, prefix rows still consume softmax mass from the existing
        tokens. The new weight for token ``i`` is ``exp(s_i) / (Z + P)``
        instead of ``exp(s_i) / Z``. Verifies the prefix rows are
        actually being concatenated.
        """
        torch.manual_seed(0)
        attn = FlashAttention(hidden_size=32, num_heads=4, dtype=torch.float16)
        x = torch.randn(2, 5, 32, dtype=torch.float16)

        out_no_prefix = attn(x).detach().clone()

        prefix_len = 3
        head_dim = 8  # 32 / 4
        zero_pk = torch.zeros(2, 4, prefix_len, head_dim, dtype=torch.float16)
        zero_pv = torch.zeros(2, 4, prefix_len, head_dim, dtype=torch.float16)
        out_with_zero_prefix = attn(x, prefix_kv=(zero_pk, zero_pv))

        mag_no = out_no_prefix.abs().mean()
        mag_zero = out_with_zero_prefix.abs().mean()
        ratio = (mag_zero / mag_no).item()
        assert 0.0 < ratio < 1.0, f"expected ratio in (0, 1), got {ratio}"
        assert ratio < 0.99, f"zero prefix should shrink output by >=1%, got {ratio}"
        assert not torch.allclose(out_no_prefix, out_with_zero_prefix, atol=1e-3)

    def test_forward_with_prefix_kv_changes_output(self):
        """Non-zero prefix → output diverges from the no-prefix path."""
        torch.manual_seed(0)
        attn = FlashAttention(hidden_size=32, num_heads=4, dtype=torch.float16)
        x = torch.randn(2, 5, 32, dtype=torch.float16)

        out_no_prefix = attn(x).detach().clone()

        prefix_len = 3
        head_dim = 8
        pk = torch.randn(2, 4, prefix_len, head_dim, dtype=torch.float16)
        pv = torch.randn(2, 4, prefix_len, head_dim, dtype=torch.float16)
        out_with_prefix = attn(x, prefix_kv=(pk, pv))

        assert not torch.allclose(out_no_prefix, out_with_prefix, atol=1e-3)

    def test_prefix_does_not_affect_output_shape(self):
        """Output shape is independent of prefix_len."""
        torch.manual_seed(0)
        attn = FlashAttention(hidden_size=32, num_heads=4, dtype=torch.float16)
        x = torch.randn(2, 5, 32, dtype=torch.float16)
        head_dim = 8
        for p_len in (1, 3, 8):
            pk = torch.randn(2, 4, p_len, head_dim, dtype=torch.float16)
            pv = torch.randn(2, 4, p_len, head_dim, dtype=torch.float16)
            out = attn(x, prefix_kv=(pk, pv))
            assert out.shape == (2, 5, 32)

    def test_prefix_kv_mismatched_shape_raises(self):
        """prefix_k and prefix_v must share the full shape."""
        attn = FlashAttention(hidden_size=32, num_heads=4, dtype=torch.float16)
        x = torch.randn(2, 5, 32, dtype=torch.float16)
        pk = torch.randn(2, 4, 3, 8, dtype=torch.float16)
        pv = torch.randn(2, 4, 5, 8, dtype=torch.float16)  # different seq dim
        with pytest.raises(ValueError, match="share shape"):
            attn(x, prefix_kv=(pk, pv))

    def test_prefix_kv_wrong_num_kv_heads_raises(self):
        """Prefix num_kv_heads must match the attention layer's num_kv_heads."""
        attn = FlashAttention(hidden_size=32, num_heads=4, dtype=torch.float16)
        x = torch.randn(2, 5, 32, dtype=torch.float16)
        pk = torch.randn(2, 3, 3, 8, dtype=torch.float16)  # 3 != num_kv_heads=4
        pv = torch.randn(2, 3, 3, 8, dtype=torch.float16)
        with pytest.raises(ValueError, match="num_kv_heads"):
            attn(x, prefix_kv=(pk, pv))

    def test_prefix_kv_wrong_head_dim_raises(self):
        """Prefix head_dim must match the attention layer's head_dim."""
        attn = FlashAttention(hidden_size=32, num_heads=4, dtype=torch.float16)
        x = torch.randn(2, 5, 32, dtype=torch.float16)
        pk = torch.randn(2, 4, 3, 16, dtype=torch.float16)  # 16 != head_dim=8
        pv = torch.randn(2, 4, 3, 16, dtype=torch.float16)
        with pytest.raises(ValueError, match="head_dim"):
            attn(x, prefix_kv=(pk, pv))

    def test_prefix_kv_wrong_batch_raises(self):
        """Prefix batch must match hidden_states batch."""
        attn = FlashAttention(hidden_size=32, num_heads=4, dtype=torch.float16)
        x = torch.randn(2, 5, 32, dtype=torch.float16)
        pk = torch.randn(3, 4, 3, 8, dtype=torch.float16)  # batch=3 != 2
        pv = torch.randn(3, 4, 3, 8, dtype=torch.float16)
        with pytest.raises(ValueError, match="batch"):
            attn(x, prefix_kv=(pk, pv))

    def test_gqa_repeats_prefix_too(self):
        """With GQA, the prefix must be repeated to match num_query_heads."""
        torch.manual_seed(0)
        attn = FlashAttention(
            hidden_size=32, num_heads=8, num_kv_heads=2, dtype=torch.float16
        )
        x = torch.randn(1, 4, 32, dtype=torch.float16)
        head_dim = 4  # 32 / 8
        pk = torch.randn(1, 2, 3, head_dim, dtype=torch.float16)
        pv = torch.randn(1, 2, 3, head_dim, dtype=torch.float16)
        out = attn(x, prefix_kv=(pk, pv))
        assert out.shape == (1, 4, 32)

    def test_prefix_dtype_is_cast_to_kv_dtype(self):
        """A float32 prefix is auto-cast to the projected K/V dtype (fp16/bf16).

        flash_attn_func requires fp16/bf16. The wrapper must cast the
        prefix silently rather than raising a dtype mismatch.
        """
        torch.manual_seed(0)
        attn = FlashAttention(hidden_size=32, num_heads=4, dtype=torch.float16)
        x = torch.randn(2, 5, 32, dtype=torch.float16)

        prefix_len = 2
        head_dim = 8
        pk = torch.randn(2, 4, prefix_len, head_dim)  # default float32
        pv = torch.randn(2, 4, prefix_len, head_dim)
        assert pk.dtype == torch.float32  # sanity

        # Should not raise even though prefix is float32 and the layer is fp16.
        out = attn(x, prefix_kv=(pk, pv))
        assert out.shape == (2, 5, 32)
        assert out.dtype == torch.float16
