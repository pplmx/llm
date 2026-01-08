#!/usr/bin/env python3
"""
Example: QLoRA fine-tuning with 4-bit quantization.

This example demonstrates how to use QLoRA for memory-efficient
fine-tuning of language models.
"""

import torch
import torch.nn as nn

from llm.core.qlora import apply_qlora, get_qlora_parameters
from llm.models.decoder import DecoderModel


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    print("=" * 50)
    print("QLoRA Example: Memory-Efficient Fine-Tuning")
    print("=" * 50)

    # Create a model
    model = DecoderModel(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        max_seq_len=128,
    )

    # Count parameters before QLoRA
    total_before, trainable_before = count_parameters(model)
    print("\n1. Original Model:")
    print(f"   Total parameters: {total_before:,}")
    print(f"   Trainable: {trainable_before:,}")

    # Apply QLoRA
    # This quantizes base weights to 4-bit and adds LoRA adapters
    apply_qlora(
        model,
        rank=8,  # LoRA rank
        alpha=16.0,  # Scaling factor (typically 2x rank)
        target_modules=["qkv_proj", "out_proj"],  # Which layers to adapt
    )

    # Count parameters after QLoRA
    total_after, trainable_after = count_parameters(model)
    print("\n2. After QLoRA:")
    print(f"   Total parameters: {total_after:,}")
    print(f"   Trainable (LoRA only): {trainable_after:,}")
    print(f"   Reduction: {100 * (1 - trainable_after / trainable_before):.1f}%")

    # Get LoRA parameters for optimizer
    lora_params = list(get_qlora_parameters(model))
    print(f"\n3. LoRA Parameter Groups: {len(lora_params)}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"   - {name}: {param.shape}")

    # Simulated training loop
    print("\n4. Simulated Training:")
    optimizer = torch.optim.AdamW(get_qlora_parameters(model), lr=1e-4)

    for step in range(3):
        # Dummy forward pass
        input_ids = torch.randint(0, 1000, (2, 32))
        logits = model(input_ids)

        # Dummy loss
        loss = logits.mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"   Step {step + 1}: loss = {loss.item():.4f}")

    print("\n5. Memory Comparison (theoretical):")
    print("   | Method     | Base Weights | Trainable | Total    |")
    print("   |------------|--------------|-----------|----------|")
    print("   | Full FT    | fp16 (100%)  | 100%      | ~2x size |")
    print("   | LoRA       | fp16 (100%)  | ~1%       | ~1x size |")
    print("   | QLoRA      | 4-bit (25%)  | ~1%       | ~0.25x   |")


if __name__ == "__main__":
    main()
