"""
Tests for ALiBi (Attention with Linear Biases) module.
"""

import pytest
import torch

from llm.core.alibi import (
    ALiBiPositionBias,
    build_alibi_bias,
    get_alibi_slopes,
)


class TestGetALiBiSlopes:
    """Tests for get_alibi_slopes function."""

    def test_power_of_2_heads(self):
        """Test slopes for power of 2 number of heads."""
        slopes = get_alibi_slopes(8)
        assert slopes.shape == (8,)
        # Slopes should be in decreasing order (geometric sequence)
        assert all(slopes[i] > slopes[i + 1] for i in range(len(slopes) - 1))

    def test_non_power_of_2_heads(self):
        """Test slopes for non-power of 2 number of heads."""
        slopes = get_alibi_slopes(6)
        assert slopes.shape == (6,)

    def test_slopes_positive(self):
        """Test that all slopes are positive."""
        for num_heads in [2, 4, 6, 8, 12, 16]:
            slopes = get_alibi_slopes(num_heads)
            assert (slopes > 0).all()


class TestBuildALiBiBias:
    """Tests for build_alibi_bias function."""

    def test_bias_shape(self, device):
        """Test bias matrix shape."""
        bias = build_alibi_bias(8, 16, device=device)
        assert bias.shape == (1, 8, 16, 16)

    def test_bias_diagonal_zero(self, device):
        """Test that diagonal (self-attention) has zero bias for squared distance."""
        bias = build_alibi_bias(4, 8, device=device)
        # Check that bias at position (i, i) for all i is 0
        for i in range(8):
            assert torch.allclose(bias[0, :, i, i], torch.zeros(4, device=device))

    def test_bias_past_negative(self, device):
        """Test that attending to past positions has bias proportional to distance."""
        bias = build_alibi_bias(4, 8, device=device)
        # Position 5 attending to position 3 should have negative bias (penalizing distance)
        # The bias formula is: distance * slope, where distance = key_pos - query_pos
        # For query=5, key=3: distance = 3 - 5 = -2, so bias = -2 * slope
        # Since slope is positive, bias is negative
        assert (bias[0, :, 5, 3] < 0).all()

    def test_bias_dtype(self, device):
        """Test bias respects dtype."""
        bias_f32 = build_alibi_bias(4, 8, device=device, dtype=torch.float32)
        bias_f16 = build_alibi_bias(4, 8, device=device, dtype=torch.float16)
        assert bias_f32.dtype == torch.float32
        assert bias_f16.dtype == torch.float16


class TestALiBiPositionBias:
    """Tests for ALiBiPositionBias class."""

    def test_init(self, device):
        """Test initialization."""
        alibi = ALiBiPositionBias(num_heads=8, device=device)
        assert alibi.num_heads == 8
        assert alibi.max_seq_len == 2048

    def test_forward_shape(self, device):
        """Test forward pass preserves shape."""
        alibi = ALiBiPositionBias(num_heads=8, device=device)
        scores = torch.randn(2, 8, 16, 16, device=device)
        result = alibi(scores)
        assert result.shape == scores.shape

    def test_forward_modifies_scores(self, device):
        """Test that forward modifies attention scores."""
        alibi = ALiBiPositionBias(num_heads=8, device=device)
        scores = torch.zeros(2, 8, 8, 8, device=device)
        result = alibi(scores)
        # Result should not be all zeros (bias added)
        assert not torch.allclose(result, scores)

    def test_get_bias(self, device):
        """Test get_bias method."""
        alibi = ALiBiPositionBias(num_heads=4, device=device)
        bias = alibi.get_bias(32)
        assert bias.shape == (1, 4, 32, 32)

    def test_caching(self, device):
        """Test that bias is cached for efficiency."""
        alibi = ALiBiPositionBias(num_heads=8, device=device)
        scores1 = torch.randn(2, 8, 16, 16, device=device)
        _ = alibi(scores1)
        cached_len1 = alibi._cached_seq_len

        # Same length should use cache
        scores2 = torch.randn(2, 8, 16, 16, device=device)
        _ = alibi(scores2)
        cached_len2 = alibi._cached_seq_len

        assert cached_len1 == cached_len2

    def test_different_seq_lens(self, device):
        """Test with different sequence lengths."""
        alibi = ALiBiPositionBias(num_heads=4, max_seq_len=64, device=device)

        for seq_len in [8, 16, 32, 64]:
            scores = torch.randn(2, 4, seq_len, seq_len, device=device)
            result = alibi(scores)
            assert result.shape == scores.shape


class TestAliBiModelIntegration:
    """DecoderModel-level ALiBi wiring (RIL — ALiBi milestone, TASK-159/ISS-190)."""

    def _model(self, use_alibi):
        torch.manual_seed(0)
        from llm.models.decoder import DecoderModel

        return DecoderModel(
            vocab_size=32,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            max_seq_len=64,
            intermediate_size=64,
            attn_impl="mha",
            use_alibi=use_alibi,
        )

    def test_builds_and_forwards(self):
        m = self._model(True)
        assert m.use_alibi is True
        assert m.alibi is not None
        assert isinstance(m.alibi, ALiBiPositionBias)
        out = m(torch.randint(0, 32, (2, 16)))
        assert out.shape == (2, 16, 32)

    def test_alibi_changes_attention_output(self):
        m_on = self._model(True)
        m_off = self._model(False)
        x = torch.randint(0, 32, (2, 16))
        with torch.no_grad():
            o_on = m_on(x)
            o_off = m_off(x)
        assert not torch.allclose(o_on, o_off, atol=1e-4), "ALiBi must change attention scores"

    def test_alibi_and_rope_are_mutually_exclusive(self):
        from llm.models.decoder import DecoderModel

        with pytest.raises(ValueError, match="mutually exclusive"):
            DecoderModel(
                vocab_size=32,
                hidden_size=32,
                num_layers=1,
                num_heads=4,
                use_alibi=True,
                use_rope=True,
                attn_impl="mha",
            )

    def test_alibi_rejected_for_non_mha_backends(self):
        from llm.models.decoder import DecoderModel

        for impl in ("flash_attn", "mla"):
            with pytest.raises((NotImplementedError, ValueError), match="mha"):
                DecoderModel(
                    vocab_size=32,
                    hidden_size=32,
                    num_layers=1,
                    num_heads=4,
                    use_alibi=True,
                    attn_impl=impl,
                )

    def test_decode_with_kv_cache(self):
        m = self._model(True)
        m.eval()
        from llm.core.kv_cache import KVCache

        cache = KVCache(max_batch_size=1, max_seq_len=64, num_kv_heads=4, head_dim=8)
        x = torch.randint(0, 32, (1, 16))
        with torch.no_grad():
            prefill, _ = m(x[:, :8], kv_caches=[cache] * 2, use_cache=True)
            decode, _ = m(x[:, 8:9], kv_caches=[cache] * 2, use_cache=True)
        assert prefill.shape == (1, 8, 32)
        assert decode.shape == (1, 1, 32)

    def test_model_config_validation(self):
        from llm.training.core.config import ModelConfig

        with pytest.raises(Exception, match="mutually exclusive"):
            ModelConfig(hidden_size=32, num_heads=4, num_layers=1, use_alibi=True, use_rope=True)
        with pytest.raises(Exception, match="mha"):
            ModelConfig(hidden_size=32, num_heads=4, num_layers=1, use_alibi=True, attn_impl="mla")


class TestSDPAWithAttnBias:
    """Direct sdpa() wrapper ALiBi bias tests (RIL — ALiBi milestone)."""

    @staticmethod
    def _sdpa(**kw):
        from llm.core.attn.sdpa import sdpa

        q, k, v = (torch.randn(2, 4, 8, 8) for _ in range(3))
        return sdpa(q, k, v, **kw)

    def test_attn_bias_without_mask(self):
        from llm.core.alibi import build_alibi_bias

        bias = build_alibi_bias(4, 8)
        out = self._sdpa(attn_bias=bias, is_causal=False)
        assert out.shape == (2, 4, 8, 8)

    def test_attn_bias_with_causal(self):
        from llm.core.alibi import build_alibi_bias

        bias = build_alibi_bias(4, 8)
        out = self._sdpa(attn_bias=bias, is_causal=True)
        assert out.shape == (2, 4, 8, 8)

    def test_attn_bias_with_bool_mask(self):
        from llm.core.alibi import build_alibi_bias

        bias = build_alibi_bias(4, 8)
        import torch as _torch

        mask = _torch.zeros(2, 1, 1, 8, dtype=_torch.bool)
        out = self._sdpa(attn_bias=bias, attn_mask=mask, is_causal=False)
        assert out.shape == (2, 4, 8, 8)
