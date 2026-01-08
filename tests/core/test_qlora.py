"""Tests for QLoRA (Quantized LoRA) module."""

import torch
import torch.nn as nn

from llm.core.qlora import (
    QLoRALinear,
    apply_qlora,
    dequantize_nf4,
    get_qlora_parameters,
    quantize_nf4,
)


class TestNF4Quantization:
    """Tests for NF4 quantization functions."""

    def test_quantize_dequantize_roundtrip(self):
        """Test that quantize -> dequantize preserves values approximately."""
        weight = torch.randn(64, 128)
        indices, scales = quantize_nf4(weight, block_size=64)

        reconstructed = dequantize_nf4(indices, scales, weight.shape, block_size=64, dtype=torch.float32)

        # NF4 is lossy but should preserve general structure
        # Correlation should be very high
        correlation = torch.corrcoef(torch.stack([weight.flatten(), reconstructed.flatten()]))[0, 1]
        assert correlation > 0.95, f"Correlation too low: {correlation}"

    def test_quantize_output_dtype(self):
        """Test that quantization produces uint8 indices."""
        weight = torch.randn(32, 32)
        indices, scales = quantize_nf4(weight)

        assert indices.dtype == torch.uint8
        assert scales.dtype == torch.float32

    def test_dequantize_output_dtype(self):
        """Test that dequantization produces requested dtype."""
        weight = torch.randn(32, 32)
        indices, scales = quantize_nf4(weight)

        fp16 = dequantize_nf4(indices, scales, weight.shape, dtype=torch.float16)
        fp32 = dequantize_nf4(indices, scales, weight.shape, dtype=torch.float32)

        assert fp16.dtype == torch.float16
        assert fp32.dtype == torch.float32


class TestQLoRALinear:
    """Tests for QLoRALinear class."""

    def test_initialization(self):
        """Test QLoRALinear initialization."""
        base = nn.Linear(64, 128)
        qlora = QLoRALinear(base, rank=4, alpha=8.0)

        assert qlora.in_features == 64
        assert qlora.out_features == 128
        assert qlora.rank == 4
        assert qlora.scaling == 2.0  # alpha / rank = 8 / 4

    def test_forward_shape(self):
        """Test that forward produces correct output shape."""
        base = nn.Linear(64, 128)
        qlora = QLoRALinear(base, rank=4)

        x = torch.randn(2, 10, 64)  # [batch, seq, in_features]
        output = qlora(x)

        assert output.shape == (2, 10, 128)

    def test_forward_with_bias(self):
        """Test forward with bias in base layer."""
        base = nn.Linear(32, 64, bias=True)
        qlora = QLoRALinear(base, rank=4)

        x = torch.randn(1, 5, 32)
        output = qlora(x)

        assert output.shape == (1, 5, 64)
        assert qlora.bias is not None

    def test_trainable_parameters(self):
        """Test that only LoRA parameters are trainable."""
        base = nn.Linear(64, 128)
        qlora = QLoRALinear(base, rank=8)

        trainable = [p for p in qlora.parameters() if p.requires_grad]
        assert len(trainable) == 2  # lora_A and lora_B

        total_trainable = sum(p.numel() for p in trainable)
        expected = 64 * 8 + 8 * 128  # in_features * rank + rank * out_features
        assert total_trainable == expected

    def test_memory_reduction(self):
        """Test that quantization reduces memory usage."""
        base = nn.Linear(1024, 2048, bias=False)
        original_bytes = base.weight.numel() * base.weight.element_size()

        qlora = QLoRALinear(base, rank=16)
        quantized_bytes = qlora.weight_indices.numel() * qlora.weight_indices.element_size()
        scales_bytes = qlora.weight_scales.numel() * qlora.weight_scales.element_size()

        # 4-bit (stored as uint8) should be 1/4 of fp32 (8x) or 1/2 of fp16 (4x)
        # Plus some overhead for scales
        total_quantized = quantized_bytes + scales_bytes
        assert total_quantized < original_bytes * 0.6  # At least 40% reduction


class TestApplyQLoRA:
    """Tests for apply_qlora function."""

    def test_apply_to_simple_model(self):
        """Test applying QLoRA to a simple model."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32, 64)
                self.fc2 = nn.Linear(64, 32)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = SimpleModel()
        apply_qlora(model, rank=4)

        assert isinstance(model.fc1, QLoRALinear)
        assert isinstance(model.fc2, QLoRALinear)

    def test_apply_with_target_modules(self):
        """Test applying QLoRA only to specific modules."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Linear(32, 32)
                self.key = nn.Linear(32, 32)
                self.value = nn.Linear(32, 32)
                self.output = nn.Linear(32, 32)

            def forward(self, x):
                return x

        model = Model()
        apply_qlora(model, rank=4, target_modules=["query", "value"])

        assert isinstance(model.query, QLoRALinear)
        assert isinstance(model.value, QLoRALinear)
        assert isinstance(model.key, nn.Linear)  # Not QLoRA
        assert isinstance(model.output, nn.Linear)  # Not QLoRA

    def test_get_qlora_parameters(self):
        """Test getting QLoRA parameters for optimizer."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(32, 64)

            def forward(self, x):
                return self.fc(x)

        model = Model()
        apply_qlora(model, rank=8)

        params = list(get_qlora_parameters(model))
        assert len(params) == 2  # lora_A and lora_B


class TestQLoRAGradients:
    """Tests for gradient flow in QLoRA."""

    def test_gradients_flow_to_lora_params(self):
        """Test that gradients flow to LoRA parameters."""
        base = nn.Linear(32, 64)
        qlora = QLoRALinear(base, rank=4)

        # Initialize lora_B with non-zero to enable gradient flow
        with torch.no_grad():
            qlora.lora_B.fill_(0.1)

        x = torch.randn(2, 5, 32)
        output = qlora(x)
        loss = output.sum()
        loss.backward()

        assert qlora.lora_A.grad is not None
        assert qlora.lora_B.grad is not None
        assert qlora.lora_A.grad.abs().sum() > 0
