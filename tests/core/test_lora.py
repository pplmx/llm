"""
Tests for LoRA (Low-Rank Adaptation) module.
"""

import pytest
import torch
import torch.nn as nn

from llm.core.lora import (
    LoRALinear,
    apply_lora,
    count_lora_parameters,
    disable_lora,
    enable_lora,
    get_lora_parameters,
    merge_lora,
    unmerge_lora,
)
from llm.models.decoder import DecoderModel


class TestLoRALinear:
    """Tests for LoRALinear class."""

    @pytest.fixture
    def base_linear(self):
        return nn.Linear(64, 128)

    @pytest.fixture
    def lora_layer(self, base_linear):
        return LoRALinear(base_linear, rank=8, alpha=16.0)

    def test_init(self, lora_layer):
        """Test LoRALinear initialization."""
        assert lora_layer.rank == 8
        assert lora_layer.alpha == 16.0
        assert lora_layer.scaling == 2.0  # 16 / 8

        # Check shapes
        assert lora_layer.lora_A.shape == (64, 8)
        assert lora_layer.lora_B.shape == (8, 128)

    def test_base_layer_frozen(self, lora_layer):
        """Test that base layer weights are frozen."""
        assert not lora_layer.base_layer.weight.requires_grad
        if lora_layer.base_layer.bias is not None:
            assert not lora_layer.base_layer.bias.requires_grad

    def test_lora_params_trainable(self, lora_layer):
        """Test that LoRA parameters are trainable."""
        assert lora_layer.lora_A.requires_grad
        assert lora_layer.lora_B.requires_grad

    def test_forward_initial(self, base_linear, lora_layer):
        """Test that initial output matches base layer (B is zero-initialized)."""
        x = torch.randn(2, 10, 64)
        base_output = base_linear(x)
        lora_output = lora_layer(x)

        # Should be identical since B is initialized to zeros
        assert torch.allclose(base_output, lora_output, atol=1e-6)

    def test_forward_after_training(self, lora_layer):
        """Test forward pass after modifying LoRA weights."""
        x = torch.randn(2, 10, 64)

        # Modify B to non-zero
        lora_layer.lora_B.data.fill_(0.1)

        output = lora_layer(x)
        base_output = lora_layer.base_layer(x)

        # Output should differ from base
        assert not torch.allclose(base_output, output)

    def test_trainable_parameters_count(self, lora_layer):
        """Test trainable parameter count."""
        expected = 64 * 8 + 8 * 128  # A + B
        assert lora_layer.trainable_parameters == expected

    def test_merge_weights(self, lora_layer):
        """Test weight merging."""
        x = torch.randn(2, 10, 64)
        lora_layer.lora_B.data.fill_(0.1)

        # Output before merge
        output_before = lora_layer(x).clone()

        # Merge and compare base layer output
        lora_layer.merge_weights()
        output_merged = lora_layer.base_layer(x)

        assert torch.allclose(output_before, output_merged, atol=1e-5)

    def test_unmerge_weights(self, lora_layer):
        """Test weight unmerging."""
        original_weight = lora_layer.base_layer.weight.clone()
        lora_layer.lora_B.data.fill_(0.1)

        lora_layer.merge_weights()
        lora_layer.unmerge_weights()

        assert torch.allclose(lora_layer.base_layer.weight, original_weight, atol=1e-5)


class TestApplyLoRA:
    """Tests for apply_lora function."""

    def test_apply_to_all_linear(self):
        """Test applying LoRA to all linear layers."""
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

        apply_lora(model, rank=4, alpha=8.0)

        # Count LoRA layers
        lora_count = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
        assert lora_count == 3

    def test_apply_to_specific_modules(self):
        """Test applying LoRA to specific modules only."""
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

        # Only apply to first linear layer (named "0")
        apply_lora(model, rank=4, target_modules=["0"])

        lora_count = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
        assert lora_count == 1

    def test_apply_to_decoder_model(self):
        """Test applying LoRA to DecoderModel."""
        model = DecoderModel(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            max_seq_len=32,
        )

        # Apply to attention QKV and output projections
        apply_lora(model, rank=8, target_modules=["qkv_proj", "out_proj"])

        lora_count = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
        assert lora_count > 0


class TestLoRAUtilities:
    """Tests for LoRA utility functions."""

    @pytest.fixture
    def model_with_lora(self):
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        apply_lora(model, rank=4, alpha=8.0)
        return model

    def test_get_lora_parameters(self, model_with_lora):
        """Test getting LoRA parameters."""
        lora_params = list(get_lora_parameters(model_with_lora))
        # 2 linear layers * 2 params (A, B) each
        assert len(lora_params) == 4

    def test_count_lora_parameters(self, model_with_lora):
        """Test counting trainable vs total parameters."""
        trainable, total = count_lora_parameters(model_with_lora)

        # Trainable should be much less than total
        assert trainable < total
        assert trainable > 0

    def test_disable_enable_lora(self, model_with_lora):
        """Test disabling and re-enabling LoRA."""
        x = torch.randn(2, 32)

        # Modify B weights
        for m in model_with_lora.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.data.fill_(0.1)

        output_enabled = model_with_lora(x).clone()

        disable_lora(model_with_lora)
        output_disabled = model_with_lora(x).clone()

        enable_lora(model_with_lora)
        output_reenabled = model_with_lora(x)

        # Disabled output should differ from enabled
        assert not torch.allclose(output_enabled, output_disabled)
        # Re-enabled should match original
        assert torch.allclose(output_enabled, output_reenabled)

    def test_merge_unmerge_full_model(self, model_with_lora):
        """Test merge and unmerge on full model."""
        x = torch.randn(2, 32)

        for m in model_with_lora.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.data.fill_(0.1)

        output_before = model_with_lora(x).clone()

        merge_lora(model_with_lora)
        unmerge_lora(model_with_lora)

        output_after = model_with_lora(x)

        assert torch.allclose(output_before, output_after, atol=1e-5)


class TestLoRATraining:
    """Tests for training with LoRA."""

    def test_training_only_lora_params(self):
        """Test that only LoRA parameters are updated during training."""
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        apply_lora(model, rank=4)

        # Store original base weights
        original_weights = {}
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                original_weights[name] = module.base_layer.weight.clone()

        # Train
        optimizer = torch.optim.Adam(get_lora_parameters(model), lr=0.01)
        x = torch.randn(4, 32)
        target = torch.randn(4, 10)
        criterion = nn.MSELoss()

        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Verify base weights unchanged
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                assert torch.allclose(module.base_layer.weight, original_weights[name])

    def test_decoder_model_training_with_lora(self):
        """Test training DecoderModel with LoRA."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = DecoderModel(
            vocab_size=100,
            hidden_size=64,
            num_layers=1,
            num_heads=2,
            max_seq_len=32,
        ).to(device)

        # Apply LoRA
        apply_lora(model, rank=4, target_modules=["qkv_proj", "out_proj", "lm_head"])

        trainable, total = count_lora_parameters(model)
        reduction = 1 - (trainable / total)
        print(f"Parameter reduction: {reduction:.1%}")

        # Train
        optimizer = torch.optim.Adam(get_lora_parameters(model), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        input_ids = torch.randint(0, 100, (2, 16), device=device)
        labels = torch.randint(0, 100, (2, 16), device=device)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, 100), labels.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Verify training happened
        assert losses[-1] != losses[0]
