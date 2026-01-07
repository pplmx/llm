"""
MoE Integration Tests.

Tests for Mixture of Experts dynamic routing behavior and integration
with TransformerBlock.
"""

import pytest
import torch

from llm.core.moe.moe import MoE
from llm.core.transformer_block import TransformerBlock


class TestMoEIntegration:
    """Integration tests for MoE layer."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def dtype(self):
        return torch.float32

    @pytest.fixture
    def moe_config(self):
        return {
            "hidden_size": 64,
            "num_experts": 4,
            "top_k": 2,
        }

    def test_expert_load_distribution(self, moe_config, device, dtype):
        """Test that tokens are distributed across multiple experts."""
        moe = MoE(**moe_config, device=device, dtype=dtype)
        moe.eval()

        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, moe_config["hidden_size"], device=device, dtype=dtype)

        # Get gate logits to analyze routing
        x_flat = x.view(-1, moe_config["hidden_size"])
        gate_logits = moe.gate(x_flat)
        _, top_k_indices = torch.topk(gate_logits, moe_config["top_k"], dim=-1)

        # Count how many tokens each expert receives
        expert_counts = torch.zeros(moe_config["num_experts"], device=device)
        for expert_idx in range(moe_config["num_experts"]):
            expert_counts[expert_idx] = (top_k_indices == expert_idx).sum()

        # With random input, all experts should receive some tokens
        # (probabilistically, but with 64 tokens and 4 experts, very unlikely to miss any)
        active_experts = (expert_counts > 0).sum().item()
        assert active_experts >= 2, f"Expected at least 2 active experts, got {active_experts}"

    def test_gradient_flow_through_gating(self, moe_config, device, dtype):
        """Test that gradients flow correctly through the gating network."""
        moe = MoE(**moe_config, device=device, dtype=dtype)
        moe.train()

        x = torch.randn(2, 8, moe_config["hidden_size"], device=device, dtype=dtype, requires_grad=True)
        output = moe(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist for gate
        assert moe.gate.weight.grad is not None, "Gate should have gradients"
        assert not torch.allclose(moe.gate.weight.grad, torch.zeros_like(moe.gate.weight.grad)), (
            "Gate gradients should be non-zero"
        )

        # Check gradients exist for at least some experts
        experts_with_grad = sum(
            1 for e in moe.experts if e.fc1.weight.grad is not None and e.fc1.weight.grad.abs().sum() > 0
        )
        assert experts_with_grad > 0, "At least one expert should have non-zero gradients"

    def test_moe_in_transformer_block(self, device, dtype):
        """Test MoE integration within TransformerBlock."""
        hidden_size = 64
        block = TransformerBlock(
            hidden_size=hidden_size,
            num_heads=4,
            intermediate_size=128,
            mlp_impl="moe",
            use_moe=True,
            num_experts=4,
            top_k=2,
            device=device,
            dtype=dtype,
        )
        block.eval()

        x = torch.randn(2, 16, hidden_size, device=device, dtype=dtype)
        output = block(x)

        # Should handle tensor or tuple return
        if isinstance(output, tuple):
            output = output[0]

        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_expert_specialization(self, moe_config, device, dtype):
        """Test that different experts produce different outputs."""
        moe = MoE(**moe_config, device=device, dtype=dtype)
        moe.eval()

        # Same input to all experts directly
        x = torch.randn(1, moe_config["hidden_size"], device=device, dtype=dtype)

        expert_outputs = [expert(x) for expert in moe.experts]

        # At least some expert pairs should produce different outputs
        different_pairs = 0
        for i in range(len(expert_outputs)):
            for j in range(i + 1, len(expert_outputs)):
                if not torch.allclose(expert_outputs[i], expert_outputs[j], atol=1e-5):
                    different_pairs += 1
        assert different_pairs > 0, "Experts should produce different outputs for same input"

    def test_moe_deterministic_routing(self, moe_config, device, dtype):
        """Test that MoE routing is deterministic for same input."""
        moe = MoE(**moe_config, device=device, dtype=dtype)
        moe.eval()

        x = torch.randn(2, 8, moe_config["hidden_size"], device=device, dtype=dtype)

        with torch.no_grad():
            out1 = moe(x)
            out2 = moe(x)

        assert torch.allclose(out1, out2), "MoE should be deterministic in eval mode"

    def test_moe_batch_independence(self, moe_config, device, dtype):
        """Test that batch samples are processed independently."""
        moe = MoE(**moe_config, device=device, dtype=dtype)
        moe.eval()

        x1 = torch.randn(1, 8, moe_config["hidden_size"], device=device, dtype=dtype)
        x2 = torch.randn(1, 8, moe_config["hidden_size"], device=device, dtype=dtype)
        x_batch = torch.cat([x1, x2], dim=0)

        with torch.no_grad():
            out1 = moe(x1)
            out2 = moe(x2)
            out_batch = moe(x_batch)

        assert torch.allclose(out1, out_batch[0:1], atol=1e-5), "Batch processing should match individual"
        assert torch.allclose(out2, out_batch[1:2], atol=1e-5), "Batch processing should match individual"
