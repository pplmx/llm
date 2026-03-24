"""
Mixture of Experts (MoE) Demo Tests

Tests MoE layer functionality.
"""

import torch

from llm.core.moe.moe import MoE


def test_moe_basic():
    """Test basic MoE layer functionality."""
    device = torch.device("cpu")
    dtype = torch.float32

    batch_size = 2
    seq_len = 10
    hidden_size = 64
    num_experts = 4
    top_k = 2

    dummy_input = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    moe_layer = MoE(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        device=device,
        dtype=dtype,
    )
    moe_layer.eval()

    output = moe_layer(dummy_input)

    assert output.shape == dummy_input.shape


def test_moe_different_top_k():
    """Test MoE with different top-k values."""
    device = torch.device("cpu")
    dtype = torch.float32

    batch_size = 2
    seq_len = 10
    hidden_size = 64
    num_experts = 8

    dummy_input = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    for top_k in [1, 2, 4]:
        moe_layer = MoE(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            device=device,
            dtype=dtype,
        )
        moe_layer.eval()

        output = moe_layer(dummy_input)
        assert output.shape == dummy_input.shape
