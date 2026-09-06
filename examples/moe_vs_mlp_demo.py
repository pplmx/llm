#!/usr/bin/env python3
"""
Example: MoE (Mixture of Experts) vs Standard MLP

This example compares the standard MLP layer with the MoE layer
using the project's actual implementations.
"""

import torch
import torch.nn as nn

from llm.core.mlp import MLP
from llm.core.moe.moe import MoE


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    print("=" * 60)
    print("MoE vs Standard MLP Comparison")
    print("=" * 60)

    hidden_size = 128
    batch_size = 2
    seq_len = 16

    # Test input
    x = torch.randn(batch_size, seq_len, hidden_size)

    # 1. Standard MLP (SwiGLU)
    # ``get_activation_layer`` has no ``"swiglu"`` name — a SwiGLU MLP is the
    # silu activation gated by ``use_glu=True`` (the same idiom tests/core/
    # test_mlp.py uses and the ADR-002 formula ``SwiGLU = Swish(xW) ⊗ xV``).
    print("\n1. Standard MLP (SwiGLU):")
    mlp = MLP(
        hidden_size=hidden_size,
        activation="silu",
        use_glu=True,
        device=x.device,
        dtype=x.dtype,
    )
    total_mlp, trainable_mlp = count_parameters(mlp)
    print(f"   Parameters: {total_mlp:,} (trainable: {trainable_mlp:,})")

    with torch.no_grad():
        output_mlp = mlp(x)
    print(f"   Output shape: {output_mlp.shape}")

    # 2. MoE Layer
    print("\n2. MoE Layer (4 experts, top-2):")
    moe = MoE(
        hidden_size=hidden_size,
        num_experts=4,
        top_k=2,
        intermediate_size=hidden_size * 4,
        device=x.device,
        dtype=x.dtype,
    )
    total_moe, trainable_moe = count_parameters(moe)
    print(f"   Parameters: {total_moe:,} (trainable: {trainable_moe:,})")
    print(f"   Parameter increase: {total_moe / total_mlp:.1f}x")

    with torch.no_grad():
        output_moe = moe(x)
        # ``MoE.forward`` returns ONLY the weighted output
        # (batch, seq, hidden) — it has no separate load return. To report
        # real routing statistics, re-run the gate over the flattened tokens
        # and count how many top-k selections each expert received.
        num_tokens = x.numel() // hidden_size
        gate_logits = moe.gate(x.view(num_tokens, hidden_size))
        expert_load = gate_logits.topk(2, dim=-1).indices.flatten().bincount(minlength=moe.num_experts)
    print(f"   Output shape: {output_moe.shape}")
    print(f"   Expert load (top-{moe.top_k} selections per token): {expert_load.tolist()}")

    # 3. Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"   MLP parameters:       {total_mlp:,}")
    print(f"   MoE parameters:       {total_moe:,}")
    print(f"   Ratio:               {total_moe / total_mlp:.1f}x")
    print("\n   MoE increases parameters but only activates top-k experts per token,")
    print("   providing more model capacity without proportional compute cost.")


if __name__ == "__main__":
    main()
