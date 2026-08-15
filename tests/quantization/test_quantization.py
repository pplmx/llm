"""Tests for Quantization module."""

import pytest
import torch
import torch.nn as nn

from llm.quantization.calibration import ActivationStats, CalibrationDataCollector
from llm.quantization.ptq import (
    QuantConfig,
    QuantizedLinear,
    compute_model_size,
    quantize_model,
)


class TestActivationStats:
    """Tests for ActivationStats."""

    def test_update_single(self):
        """Test stats update with single tensor."""
        stats = ActivationStats(name="test")

        tensor = torch.randn(10, 20)
        stats.update(tensor)

        assert stats.num_samples == 200
        assert stats.min_val <= tensor.min().item()
        assert stats.max_val >= tensor.max().item()

    def test_update_multiple(self):
        """Test stats update with multiple tensors."""
        stats = ActivationStats(name="test")

        for _ in range(5):
            stats.update(torch.randn(10, 20))

        assert stats.num_samples == 1000

    def test_compute_scale_symmetric(self):
        """Test symmetric scale computation."""
        stats = ActivationStats(name="test")
        stats.abs_max = 127.0

        scale = stats.compute_scale(bits=8, symmetric=True)

        assert scale == pytest.approx(1.0, rel=0.01)

    def test_compute_scale_asymmetric(self):
        """Test asymmetric scale computation."""
        stats = ActivationStats(name="test")
        stats.min_val = 0.0
        stats.max_val = 255.0

        scale = stats.compute_scale(bits=8, symmetric=False)

        assert scale == pytest.approx(1.0, rel=0.01)


class TestCalibrationDataCollector:
    """Tests for CalibrationDataCollector."""

    def test_register_hooks(self):
        """Test hook registration."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

        collector = CalibrationDataCollector(model)
        collector.register_hooks()

        assert len(collector.hooks) == 2  # Two Linear layers
        assert "0" in collector.stats
        assert "2" in collector.stats

        collector.remove_hooks()

    def test_collect_stats(self):
        """Test statistics collection."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        collector = CalibrationDataCollector(model)
        collector.register_hooks()

        # Simulate forward pass
        x = torch.randn(4, 10)
        model(x)

        assert collector.stats["0"].num_samples > 0
        assert collector.stats["2"].num_samples > 0

        collector.remove_hooks()

    def test_context_manager(self):
        """Test context manager usage."""
        model = nn.Linear(10, 10)

        with CalibrationDataCollector(model) as collector:
            collector.register_hooks()
            model(torch.randn(4, 10))

        # Hooks should be removed
        assert len(collector.hooks) == 0


class TestQuantConfig:
    """Tests for QuantConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = QuantConfig()

        assert config.bits == 8
        assert config.symmetric is True
        assert config.per_channel is False

    def test_invalid_bits(self):
        """Test that invalid bits raise error."""
        with pytest.raises(ValueError, match="Unsupported bit width"):
            QuantConfig(bits=16)


class TestQuantizedLinear:
    """Tests for QuantizedLinear."""

    def test_from_linear(self):
        """Test conversion from Linear."""
        linear = nn.Linear(10, 20)
        quant_linear = QuantizedLinear.from_linear(linear)

        assert quant_linear.in_features == 10
        assert quant_linear.out_features == 20
        assert quant_linear.weight_quantized.dtype == torch.int8

    def test_forward_pass(self):
        """Test forward pass produces output."""
        linear = nn.Linear(10, 20)
        quant_linear = QuantizedLinear.from_linear(linear)

        x = torch.randn(4, 10)
        output = quant_linear(x)

        assert output.shape == (4, 20)

    def test_per_channel_quantization(self):
        """Test per-channel quantization."""
        config = QuantConfig(per_channel=True)
        linear = nn.Linear(10, 20)
        quant_linear = QuantizedLinear.from_linear(linear, config)

        assert quant_linear.weight_scale.shape == (20,)

    def test_per_tensor_quantization(self):
        """Test per-tensor quantization."""
        config = QuantConfig(per_channel=False)
        linear = nn.Linear(10, 20)
        quant_linear = QuantizedLinear.from_linear(linear, config)

        assert quant_linear.weight_scale.shape == (1,)

    def test_forward_accepts_fp16_bf16_input(self):
        """fp16/bf16 inputs must not crash (regression for ISS-018)."""
        linear = nn.Linear(10, 20)
        quant_linear = QuantizedLinear.from_linear(linear)

        x32 = torch.randn(4, 10)
        ref = quant_linear(x32)
        for x in (x32.half(), x32.bfloat16()):
            out = quant_linear(x)
            assert out.dtype == torch.float32
            assert torch.allclose(out, ref, atol=1e-1)


class TestQuantizeModel:
    """Tests for model quantization."""

    def test_quantize_model(self):
        """Test full model quantization."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

        quant_model = quantize_model(model)

        # Check layers are replaced
        assert isinstance(quant_model[0], QuantizedLinear)
        assert isinstance(quant_model[2], QuantizedLinear)

    def test_quantize_preserves_output(self):
        """Test that quantized model produces similar output."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        x = torch.randn(4, 10)
        original_output = model(x)

        quant_model = quantize_model(model)
        quant_output = quant_model(x)

        # Output shape should match
        assert quant_output.shape == original_output.shape

        # Output should be similar (not exact due to quantization)
        # We allow some tolerance due to quantization error
        assert torch.allclose(quant_output, original_output, atol=0.5)

    def test_compute_model_size(self):
        """Test model size computation."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 10),
        )

        size_info = compute_model_size(model)

        assert size_info["total_params"] > 0
        assert size_info["total_bytes"] > 0
        assert size_info["quantized_layers"] == 0

        quant_model = quantize_model(model)
        quant_size_info = compute_model_size(quant_model)

        assert quant_size_info["quantized_layers"] == 2
        # Quantized model should be smaller
        assert quant_size_info["total_bytes"] < size_info["total_bytes"]

    def test_compute_model_size_counts_gptq_quantized_layers(self):
        """compute_model_size must not report all-zero stats for a GPTQ-
        quantized model (regression for ISS-020).

        GPTQ replaces ``nn.Linear`` with ``GPTQQuantizedLinear``; before the
        fix the stats loop only recognised simple-PTQ ``QuantizedLinear`` and
        ``nn.Linear``, so a ``quantize_model_gptq`` result reported 0 params,
        0 bytes and 0 quantized layers even though the layers were quantized.
        """
        from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

        # Capture the fp32 size BEFORE quantize_model_gptq mutates the model
        # in place (it replaces nn.Linear modules with GPTQQuantizedLinear).
        plain_model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 8))
        plain_size = compute_model_size(plain_model)

        model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 8))
        calib = torch.randn(4, 8)
        quant_model = quantize_model_gptq(
            model,
            iter([calib]),
            config=GPTQConfig(bits=4, group_size=-1, sym=True),
        )

        size_info = compute_model_size(quant_model)
        assert size_info["quantized_layers"] == 2
        assert size_info["total_params"] > 0
        assert size_info["total_bytes"] > 0
        assert size_info["size_mb"] > 0.0
        # Packed int4 weights (+ int8 storage) are strictly smaller than the
        # fp32 original parameters.
        assert size_info["total_bytes"] < plain_size["total_bytes"]

    def test_compute_model_size_counts_true_weights_for_4bit(self):
        """``total_params`` must count true weights, not packed bytes, for
        4-bit GPTQ/AWQ (regression for ISS-94).

        ``weight_packed`` is an int8 tensor holding TWO int4 weights per byte,
        so ``packed.numel()`` undercounts the parameter count by 2x for
        bits=4. ``total_bytes`` (actual packed storage) is unaffected.
        """
        from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

        # Capturing the fp32 size BEFORE quantize_model_gptq mutates the
        # model in place (it replaces nn.Linear modules with GPTQQuantizedLinear).
        plain_model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 8))
        true_params = sum(p.numel() for p in plain_model.parameters() if p.ndim == 2)
        calib = torch.randn(4, 8)

        # bits=4: two int4 weights packed per int8 byte → params = 2x bytes.
        model_4 = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 8))
        quant_4 = quantize_model_gptq(
            model_4,
            iter([calib]),
            config=GPTQConfig(bits=4, group_size=-1, sym=True),
        )
        info_4 = compute_model_size(quant_4)
        assert info_4["total_params"] == true_params, (
            f"bits=4 total_params {info_4['total_params']} != true {true_params} "
            "(packed bytes undercount int4 weights by 2x)"
        )

        # bits=8 stores one weight per byte → already counted correctly.
        model_8 = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 8))
        quant_8 = quantize_model_gptq(
            model_8,
            iter([calib]),
            config=GPTQConfig(bits=8, group_size=-1, sym=True),
        )
        info_8 = compute_model_size(quant_8)
        assert info_8["total_params"] == true_params

    def test_from_linear_rejects_asymmetric(self):
        """Asymmetric simple-PTQ was a silent no-op (zero point never
        computed); it must fail fast instead, mirroring GPTQQuantizedLinear."""
        linear = torch.nn.Linear(4, 4)
        with pytest.raises(NotImplementedError, match="Asymmetric"):
            QuantizedLinear.from_linear(linear, QuantConfig(bits=8, symmetric=False))
