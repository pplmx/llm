"""End-to-end tests for GPTQ integration on full models."""

import pytest
import torch
import torch.nn as nn


class TwoLayerMLP(nn.Module):
    """Tiny model for GPTQ end-to-end testing."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_quantize_model_gptq_replaces_all_linear_layers():
    """quantize_model_gptq converts every nn.Linear to GPTQQuantizedLinear."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    model = TwoLayerMLP(hidden=16)
    calib = [torch.randn(8, 16) for _ in range(4)]

    quantized = quantize_model_gptq(model, iter(calib), GPTQConfig())

    linear_count = sum(1 for _ in quantized.modules() if isinstance(_, nn.Linear))
    gptq_count = sum(1 for _ in quantized.modules() if isinstance(_, GPTQQuantizedLinear))

    assert linear_count == 0, f"Expected 0 nn.Linear, found {linear_count}"
    assert gptq_count == 2, f"Expected 2 GPTQQuantizedLinear (fc1, fc2), found {gptq_count}"


def test_quantize_model_gptq_preserves_forward_contract():
    """Quantized model accepts same input shape and returns same output shape."""
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    torch.manual_seed(123)
    model = TwoLayerMLP(hidden=16)
    calib = [torch.randn(8, 16) for _ in range(4)]

    quantized = quantize_model_gptq(model, iter(calib), GPTQConfig())

    x = torch.randn(2, 16)
    out = quantized(x)
    assert out.shape == (2, 16)


def test_quantize_model_gptq_rejects_already_quantized():
    """Passing a model with GPTQQuantizedLinear raises ValueError."""
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    model = TwoLayerMLP(hidden=16)
    calib = [torch.randn(8, 16) for _ in range(4)]
    quantized = quantize_model_gptq(model, iter(calib), GPTQConfig())

    with pytest.raises(ValueError, match="already GPTQ-quantized"):
        quantize_model_gptq(quantized, iter(calib), GPTQConfig())


def test_quantize_model_gptq_no_linear_raises():
    """Model with no nn.Linear raises ValueError."""
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    model = nn.Sequential(nn.GELU(), nn.GELU())  # no Linear
    with pytest.raises(ValueError, match=r"no nn\.Linear"):
        quantize_model_gptq(model, iter([torch.randn(4, 8)]), GPTQConfig())


def test_target_modules_filters_correctly():
    """target_modules restricts which Linear layers get quantized."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    model = TwoLayerMLP(hidden=16)
    calib = [torch.randn(8, 16) for _ in range(4)]

    # Only quantize fc1, leave fc2 as nn.Linear
    quantized = quantize_model_gptq(model, iter(calib), GPTQConfig(), target_modules=["fc1"])

    fc1_layer = quantized.fc1
    fc2_layer = quantized.fc2

    assert isinstance(fc1_layer, GPTQQuantizedLinear)
    assert isinstance(fc2_layer, nn.Linear)
    assert not isinstance(fc2_layer, GPTQQuantizedLinear)


def test_quantize_model_with_collector_works():
    """quantize_model_with_collector reuses any iterator-yielding collector."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear
    from llm.quantization.gptq import GPTQConfig, quantize_model_with_collector

    # Duck-typed collector (matches CalibrationDataCollector protocol: __iter__)
    class MockCollector:
        def __init__(self, batches):
            self.batches = batches

        def __iter__(self):
            return iter(self.batches)

    model = TwoLayerMLP(hidden=16)
    batches = [torch.randn(8, 16) for _ in range(3)]
    collector = MockCollector(batches)

    quantized = quantize_model_with_collector(model, collector, n_samples=3, config=GPTQConfig())

    gptq_count = sum(1 for _ in quantized.modules() if isinstance(_, GPTQQuantizedLinear))
    assert gptq_count == 2


def test_quantize_model_with_collector_respects_n_samples():
    """n_samples caps how many batches are consumed from the collector."""
    from llm.quantization.gptq import GPTQConfig, quantize_model_with_collector

    yielded = {"count": 0}

    def collector():
        for b in [torch.randn(8, 16) for _ in range(5)]:
            yielded["count"] += 1
            yield b

    model = TwoLayerMLP(hidden=16)
    quantize_model_with_collector(model, collector(), n_samples=2, config=GPTQConfig())

    # Only the first 2 batches should have been pulled from the generator.
    assert yielded["count"] == 2, f"Expected 2 batches consumed, got {yielded['count']}"


def test_quantize_model_with_collector_propagates_target_modules():
    """target_modules filter is forwarded to quantize_model_gptq."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear
    from llm.quantization.gptq import GPTQConfig, quantize_model_with_collector

    class MockCollector:
        def __init__(self, batches):
            self.batches = batches

        def __iter__(self):
            return iter(self.batches)

    model = TwoLayerMLP(hidden=16)
    collector = MockCollector([torch.randn(8, 16) for _ in range(2)])

    quantized = quantize_model_with_collector(
        model, collector, n_samples=2, config=GPTQConfig(), target_modules=["fc1"]
    )

    assert isinstance(quantized.fc1, GPTQQuantizedLinear)
    assert isinstance(quantized.fc2, nn.Linear)


# === Coverage gap tests: 8-bit e2e, target_modules no match, empty calib, n_samples <= 0, device kwarg ===


def test_quantize_model_gptq_8bit_e2e_works():
    """bits=8 end-to-end produces GPTQQuantizedLinear with int8 (un-packed) storage."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    model = TwoLayerMLP(hidden=16)
    calib = [torch.randn(8, 16) for _ in range(3)]

    quantized = quantize_model_gptq(model, iter(calib), GPTQConfig(bits=8, group_size=-1))

    assert isinstance(quantized.fc1, GPTQQuantizedLinear)
    assert isinstance(quantized.fc2, GPTQQuantizedLinear)
    assert quantized.fc1.bits == 8
    assert quantized.fc2.bits == 8
    # 8-bit storage: weight_packed has the same number of elements as the weight (no half-size)
    assert quantized.fc1.weight_packed.numel() == 16 * 32  # out x in for fc1 (16->32)
    # Forward still works
    out = quantized(torch.randn(2, 16))
    assert out.shape == (2, 16)


def test_quantize_model_gptq_target_modules_no_match_raises_with_available():
    """target_modules that match nothing raises ValueError naming available layers."""
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    model = TwoLayerMLP(hidden=16)
    calib = [torch.randn(8, 16) for _ in range(2)]

    with pytest.raises(ValueError, match=r"matched no nn\.Linear") as exc:
        quantize_model_gptq(model, iter(calib), GPTQConfig(), target_modules=["nonexistent.layer"])

    # Error message names at least one available layer for the user to copy
    assert "fc1" in str(exc.value) or "fc2" in str(exc.value)


def test_quantize_model_gptq_empty_calib_raises():
    """calib_iter with zero batches raises ValueError before any model mutation."""
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    model = TwoLayerMLP(hidden=16)

    with pytest.raises(ValueError, match="empty"):
        quantize_model_gptq(model, iter([]), GPTQConfig())


def test_quantize_model_with_collector_rejects_non_positive_n_samples():
    """n_samples <= 0 raises ValueError — guards against divide-by-zero / no-op confusion."""
    from llm.quantization.gptq import GPTQConfig, quantize_model_with_collector

    def collector():
        yield torch.randn(8, 16)

    model = TwoLayerMLP(hidden=16)

    with pytest.raises(ValueError, match="n_samples must be positive"):
        quantize_model_with_collector(model, collector(), n_samples=0, config=GPTQConfig())

    with pytest.raises(ValueError, match="n_samples must be positive"):
        quantize_model_with_collector(model, collector(), n_samples=-3, config=GPTQConfig())


def test_quantize_model_gptq_device_kwarg_relocates_model():
    """device='cpu' (or any valid device) is forwarded to model.to() before calibration."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    model = TwoLayerMLP(hidden=16)
    calib = [torch.randn(8, 16) for _ in range(2)]

    # Explicit device kwarg is accepted (idempotent on CPU model)
    quantized = quantize_model_gptq(model, iter(calib), GPTQConfig(), device="cpu")

    assert isinstance(quantized.fc1, GPTQQuantizedLinear)
    # All parameters and buffers live on CPU
    for p in quantized.parameters():
        assert p.device.type == "cpu"
    for b in quantized.buffers():
        assert b.device.type == "cpu"


def test_quantize_model_gptq_4bit_per_channel_storage_branch():
    """4-bit + group_size=-1 (per-channel) end-to-end: hits the per-channel storage branch."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    model = TwoLayerMLP(hidden=16)
    calib = [torch.randn(8, 16) for _ in range(3)]

    quantized = quantize_model_gptq(model, iter(calib), GPTQConfig(bits=4, group_size=-1))

    assert isinstance(quantized.fc1, GPTQQuantizedLinear)
    assert quantized.fc1.bits == 4
    assert quantized.fc1.group_size == -1
    # Per-channel: scales shape [out_features, 1]
    assert quantized.fc1.scales.shape == (32, 1)  # fc1: in=16, out=32
    assert quantized.fc2.scales.shape == (16, 1)  # fc2: in=32, out=16


def test_quantize_model_gptq_8bit_per_group_storage_branch():
    """8-bit + group_size=4 (per-group) end-to-end: hits the 8-bit per-group storage branch."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    model = TwoLayerMLP(hidden=16)
    calib = [torch.randn(8, 16) for _ in range(3)]

    quantized = quantize_model_gptq(model, iter(calib), GPTQConfig(bits=8, group_size=4))

    assert isinstance(quantized.fc1, GPTQQuantizedLinear)
    assert quantized.fc1.bits == 8
    # Per-group: scales shape [out_features, in_features // group_size]
    assert quantized.fc1.scales.shape == (32, 16 // 4)  # fc1: in=16, group_size=4
    # 8-bit storage: same number of elements as the weight (no half-size)
    assert quantized.fc1.weight_packed.numel() == 32 * 16


def test_quantize_model_gptq_falls_back_when_model_forward_fails():
    """If model forward raises on the calibration batch, fall back to direct layer calls."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    class AlwaysFailingForward(nn.Module):
        """Model whose forward always raises — forces the except branch."""

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 4)

        def forward(self, x):
            raise RuntimeError("intentional failure to force fallback path")

    model = AlwaysFailingForward()
    calib = [torch.randn(8, 8) for _ in range(2)]

    quantized = quantize_model_gptq(model, iter(calib), GPTQConfig())

    assert isinstance(quantized.fc1, GPTQQuantizedLinear)


def test_quantize_model_gptq_falls_back_when_forward_skips_target_layers():
    """If model forward succeeds but never calls target layers, fall back to direct calls."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear
    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    class ForwardSkipsLinear(nn.Module):
        """Model whose forward never invokes self.fc1 — hooks capture nothing."""

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 4)
            self.register_buffer("zeros", torch.zeros(4))

        def forward(self, x):
            # Return a constant — never touch fc1
            return self.zeros.unsqueeze(0).expand(x.shape[0], -1)

    model = ForwardSkipsLinear()
    calib = [torch.randn(8, 8) for _ in range(2)]

    quantized = quantize_model_gptq(model, iter(calib), GPTQConfig())

    # The fallback path ran — fc1 was quantized via direct call
    assert isinstance(quantized.fc1, GPTQQuantizedLinear)
