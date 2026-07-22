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
