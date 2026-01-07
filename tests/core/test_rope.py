"""
Tests for Rotary Position Embedding (RoPE) module.
"""

import pytest
import torch

from llm.core.rope import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
    get_rope_scaling_factor,
    rotate_half,
)


class TestRotateHalf:
    """Tests for rotate_half function."""

    def test_rotate_half_shape(self):
        """Test that rotate_half preserves shape."""
        x = torch.randn(2, 4, 8, 64)
        result = rotate_half(x)
        assert result.shape == x.shape

    def test_rotate_half_values(self):
        """Test rotate_half values are correct."""
        x = torch.tensor([[1, 2, 3, 4]])
        result = rotate_half(x)
        expected = torch.tensor([[-3, -4, 1, 2]])
        assert torch.allclose(result, expected)


class TestApplyRotaryPosEmb:
    """Tests for apply_rotary_pos_emb function."""

    def test_apply_rotary_shape(self):
        """Test that apply_rotary_pos_emb preserves shape."""
        x = torch.randn(2, 4, 8, 64)
        cos = torch.ones(1, 1, 8, 64)
        sin = torch.zeros(1, 1, 8, 64)
        result = apply_rotary_pos_emb(x, cos, sin)
        assert result.shape == x.shape

    def test_identity_with_zero_sin(self):
        """Test that zero sin gives identity (cos=1)."""
        x = torch.randn(2, 4, 8, 64)
        cos = torch.ones(1, 1, 8, 64)
        sin = torch.zeros(1, 1, 8, 64)
        result = apply_rotary_pos_emb(x, cos, sin)
        assert torch.allclose(result, x)


class TestRotaryPositionEmbedding:
    """Tests for RotaryPositionEmbedding class."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_init_default(self, device):
        """Test default initialization."""
        rope = RotaryPositionEmbedding(dim=64, device=device)
        assert rope.dim == 64
        assert rope.max_seq_len == 2048
        assert rope.base == 10000.0
        assert rope.scaling_type is None
        assert rope.scaling_factor == 1.0

    def test_init_with_scaling(self, device):
        """Test initialization with scaling."""
        rope = RotaryPositionEmbedding(
            dim=64,
            scaling_type="linear",
            scaling_factor=2.0,
            device=device,
        )
        assert rope.scaling_type == "linear"
        assert rope.scaling_factor == 2.0

    def test_forward_shape(self, device):
        """Test forward pass shape."""
        rope = RotaryPositionEmbedding(dim=64, device=device)
        q = torch.randn(2, 4, 16, 64, device=device)
        k = torch.randn(2, 4, 16, 64, device=device)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_forward_different_seq_lens(self, device):
        """Test forward with different sequence lengths."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=128, device=device)

        for seq_len in [8, 32, 64, 128]:
            q = torch.randn(2, 4, seq_len, 64, device=device)
            k = torch.randn(2, 4, seq_len, 64, device=device)
            q_rot, k_rot = rope(q, k)
            assert q_rot.shape == q.shape

    def test_forward_with_position_ids(self, device):
        """Test forward with explicit position ids."""
        rope = RotaryPositionEmbedding(dim=64, device=device)
        q = torch.randn(2, 4, 8, 64, device=device)
        k = torch.randn(2, 4, 8, 64, device=device)
        position_ids = torch.arange(8, device=device).unsqueeze(0).expand(2, -1)

        q_rot, k_rot = rope(q, k, position_ids)
        assert q_rot.shape == q.shape

    def test_linear_scaling(self, device):
        """Test linear scaling produces different results."""
        rope_no_scale = RotaryPositionEmbedding(dim=64, device=device)
        rope_scaled = RotaryPositionEmbedding(
            dim=64,
            scaling_type="linear",
            scaling_factor=2.0,
            device=device,
        )

        q = torch.randn(2, 4, 16, 64, device=device)
        k = torch.randn(2, 4, 16, 64, device=device)

        q_no_scale, k_no_scale = rope_no_scale(q, k)
        q_scaled, k_scaled = rope_scaled(q, k)

        # Results should differ
        assert not torch.allclose(q_no_scale, q_scaled)

    def test_ntk_scaling(self, device):
        """Test NTK-aware scaling."""
        rope_ntk = RotaryPositionEmbedding(
            dim=64,
            scaling_type="ntk",
            scaling_factor=2.0,
            device=device,
        )

        q = torch.randn(2, 4, 16, 64, device=device)
        k = torch.randn(2, 4, 16, 64, device=device)

        q_rot, k_rot = rope_ntk(q, k)
        assert q_rot.shape == q.shape

    def test_dynamic_scaling(self, device):
        """Test dynamic scaling with extended context."""
        rope = RotaryPositionEmbedding(
            dim=64,
            max_seq_len=64,
            scaling_type="dynamic",
            device=device,
        )

        # Test with sequence longer than max_seq_len
        q = torch.randn(2, 4, 128, 64, device=device)
        k = torch.randn(2, 4, 128, 64, device=device)

        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape


class TestGetRopeScalingFactor:
    """Tests for get_rope_scaling_factor function."""

    def test_no_scaling_within_context(self):
        """Test no scaling when within trained context."""
        factor = get_rope_scaling_factor(1024, 2048, "linear")
        assert factor == 1.0

    def test_linear_scaling(self):
        """Test linear scaling factor."""
        factor = get_rope_scaling_factor(4096, 2048, "linear")
        assert factor == 2.0

    def test_dynamic_scaling(self):
        """Test dynamic scaling factor."""
        factor = get_rope_scaling_factor(4096, 2048, "dynamic")
        assert factor == pytest.approx(1.414, rel=0.01)  # sqrt(2)
