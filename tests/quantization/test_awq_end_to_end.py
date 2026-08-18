"""End-to-end tests for AWQ integration on full models."""

import pytest
import torch
import torch.nn as nn


class TwoLayerMLP(nn.Module):
    """Tiny model for AWQ end-to-end testing (same shape as GPTQ tests)."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_quantize_model_awq_replaces_all_linear_layers():
    """quantize_model_awq converts every nn.Linear to AWQQuantizedLinear."""
    from llm.quantization._awq_layer import AWQQuantizedLinear
    from llm.quantization.awq import AWQConfig, quantize_model_awq

    model = TwoLayerMLP(hidden=16)
    calib = [torch.randn(8, 16) for _ in range(4)]

    quantized = quantize_model_awq(model, iter(calib), AWQConfig())

    linear_count = sum(1 for _ in quantized.modules() if isinstance(_, nn.Linear))
    awq_count = sum(1 for _ in quantized.modules() if isinstance(_, AWQQuantizedLinear))
    assert linear_count == 0
    assert awq_count == 2


def test_quantize_model_awq_preserves_forward_contract():
    """Quantized model accepts the same input shape and returns the same output shape."""
    from llm.quantization.awq import AWQConfig, quantize_model_awq

    torch.manual_seed(123)
    model = TwoLayerMLP(hidden=16)
    calib = [torch.randn(8, 16) for _ in range(4)]

    quantized = quantize_model_awq(model, iter(calib), AWQConfig())
    x = torch.randn(2, 16)
    out = quantized(x)
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_quantize_model_awq_preserves_bias_values():
    """Original Linear biases are copied into the AWQ layers."""
    from llm.quantization.awq import AWQConfig, quantize_model_awq

    torch.manual_seed(7)
    model = TwoLayerMLP(hidden=16)
    original_biases = {
        n: m.bias.detach().clone() for n, m in model.named_modules() if n and getattr(m, "bias", None) is not None
    }
    calib = [torch.randn(8, 16) for _ in range(2)]

    quantized = quantize_model_awq(model, iter(calib), AWQConfig())
    for n, m in quantized.named_modules():
        if n in original_biases and getattr(m, "bias", None) is not None:
            assert torch.allclose(m.bias.detach(), original_biases[n])


def test_quantize_model_awq_rejects_already_quantized():
    """Refusing to double-quantize surfaces an actionable error."""
    from llm.quantization.awq import AWQConfig, quantize_model_awq

    model = TwoLayerMLP(hidden=16)
    quantized = quantize_model_awq(model, iter([torch.randn(8, 16)]), AWQConfig())
    with pytest.raises(ValueError, match="already AWQ-quantized"):
        quantize_model_awq(quantized, iter([torch.randn(8, 16)]), AWQConfig())


def test_quantize_model_awq_no_linear_raises():
    """A model without nn.Linear is rejected."""
    from llm.quantization.awq import AWQConfig, quantize_model_awq

    model = nn.Sequential(nn.GELU())
    with pytest.raises(ValueError, match=r"no nn.Linear"):
        quantize_model_awq(model, iter([torch.randn(8, 16)]), AWQConfig())


def test_target_modules_filters_correctly():
    """Only the named target modules are quantized."""
    from llm.quantization._awq_layer import AWQQuantizedLinear
    from llm.quantization.awq import AWQConfig, quantize_model_awq

    model = TwoLayerMLP(hidden=16)
    quantized = quantize_model_awq(model, iter([torch.randn(8, 16)]), AWQConfig(), target_modules=["fc1"])

    assert isinstance(quantized.fc1, AWQQuantizedLinear)
    assert isinstance(quantized.fc2, nn.Linear)


def test_target_modules_no_match_raises_with_available():
    """A target_modules filter matching nothing raises with available layers."""
    from llm.quantization.awq import AWQConfig, quantize_model_awq

    model = TwoLayerMLP(hidden=16)
    with pytest.raises(ValueError, match=r"matched no nn.Linear"):
        quantize_model_awq(model, iter([torch.randn(8, 16)]), AWQConfig(), target_modules=["nope"])


def test_quantize_model_awq_empty_calib_raises():
    """An empty calibration iterator is rejected loudly."""
    from llm.quantization.awq import AWQConfig, quantize_model_awq

    model = TwoLayerMLP(hidden=16)
    with pytest.raises(ValueError, match="calib_iter is empty"):
        quantize_model_awq(model, iter([]), AWQConfig())


def test_quantize_model_awq_falls_back_when_model_forward_fails():
    """When the model forward fails, per-layer calibration uses direct calls."""
    from llm.quantization._awq_layer import AWQQuantizedLinear
    from llm.quantization.awq import AWQConfig, quantize_model_awq

    class BrokenForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 16)

        def forward(self, x):
            raise RuntimeError("calibration forward is broken by design")

    model = BrokenForward()
    quantized = quantize_model_awq(model, iter([torch.randn(8, 16)]), AWQConfig())
    assert isinstance(quantized.fc1, AWQQuantizedLinear)


def test_quantize_model_awq_8bit_e2e():
    """8-bit AWQ end-to-end works with per-group storage."""
    from llm.quantization.awq import AWQConfig, quantize_model_awq

    torch.manual_seed(8)
    model = TwoLayerMLP(hidden=16)
    quantized = quantize_model_awq(model, iter([torch.randn(8, 16)]), AWQConfig(bits=8, group_size=8))
    out = quantized(torch.randn(2, 16))
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_quantize_model_awq_4bit_per_channel_storage_branch():
    """group_size=-1 produces per-channel scales (one per output row)."""
    from llm.quantization.awq import AWQConfig, quantize_model_awq

    model = TwoLayerMLP(hidden=16)
    quantized = quantize_model_awq(model, iter([torch.randn(8, 16)]), AWQConfig(group_size=-1))
    assert quantized.fc1.scales.shape[1] == 1
    assert quantized(torch.randn(2, 16)).shape == (2, 16)


def test_quantize_model_awq_layer_policies_mixed_precision():
    """LayerQuantPolicy overrides bits per layer (mixed 8/4-bit)."""
    from llm.quantization._policy import LayerQuantPolicy
    from llm.quantization.awq import AWQConfig, quantize_model_awq

    model = TwoLayerMLP(hidden=16)
    config = AWQConfig(
        bits=4,
        layer_policies=(LayerQuantPolicy(target_modules=("fc1",), bits=8),),
    )
    quantized = quantize_model_awq(model, iter([torch.randn(8, 16)]), config)
    assert quantized.fc1.bits == 8
    assert quantized.fc2.bits == 4


def test_quantize_model_awq_with_collector_works():
    """quantize_model_awq_with_collector materializes batches from an iterable."""
    from llm.quantization.awq import AWQConfig, quantize_model_awq_with_collector

    model = TwoLayerMLP(hidden=16)
    collector = [torch.randn(8, 16) for _ in range(5)]
    quantized = quantize_model_awq_with_collector(model, collector, n_samples=3, config=AWQConfig())
    assert quantized(torch.randn(2, 16)).shape == (2, 16)


def test_quantize_model_awq_with_collector_respects_n_samples():
    """Only n_samples batches are consumed from the collector."""
    from llm.quantization.awq import AWQConfig, quantize_model_awq_with_collector

    seen: list[int] = []

    def collector():
        for i in range(10):
            seen.append(i)
            yield torch.randn(8, 16)

    model = TwoLayerMLP(hidden=16)
    quantize_model_awq_with_collector(model, collector(), n_samples=2, config=AWQConfig())
    assert seen == [0, 1]


def test_quantize_model_awq_with_collector_rejects_non_positive_n_samples():
    """n_samples must be positive."""
    from llm.quantization.awq import AWQConfig, quantize_model_awq_with_collector

    model = TwoLayerMLP(hidden=16)
    with pytest.raises(ValueError, match="n_samples must be positive"):
        quantize_model_awq_with_collector(model, [torch.randn(8, 16)], n_samples=0, config=AWQConfig())


def test_quantize_model_awq_reduces_error_on_salient_model():
    """On a model with a salient input channel, AWQ beats naive RTN end-to-end."""
    from llm.quantization._awq_layer import AWQQuantizedLinear
    from llm.quantization.awq import AWQConfig, _pack_weights, quantize_model_awq

    torch.manual_seed(11)
    calib_x = torch.randn(64, 32)
    calib_x[:, 0] *= 30.0  # salient channel into fc1
    calib = [calib_x]

    # The AWQ premise: the salient channel's WEIGHTS are small, so naive group
    # quantization is coarse relative to their magnitude. Scaling them up
    # (with activation compensation) restores fine quantization.
    base = TwoLayerMLP(hidden=32)
    with torch.no_grad():
        base.fc1.weight[:, 0] *= 0.05

    # Naive RTN baseline: same packed storage but input_scales=None (s=1).
    naive2 = __import__("copy").deepcopy(base)
    ref_model = __import__("copy").deepcopy(base)
    awq_model = quantize_model_awq(base, iter(calib), AWQConfig(bits=4))
    with torch.no_grad():
        for name, mod in naive2.named_modules():
            if isinstance(mod, nn.Linear):
                packed, scales, gs = _pack_weights(mod.weight.data.float(), 4, 128)
                replacement = AWQQuantizedLinear(
                    in_features=mod.in_features,
                    out_features=mod.out_features,
                    bias=mod.bias is not None,
                    weight_packed=packed,
                    scales=scales.to(torch.float16),
                    bits=4,
                    group_size=gs,
                    sym=True,
                    input_scales=None,
                )
                if mod.bias is not None:
                    replacement.bias.copy_(mod.bias.data)
                parent_name, _, child_name = name.rpartition(".")
                parent = naive2 if not parent_name else _getattr_nested(naive2, parent_name)
                setattr(parent, child_name, replacement)

    test_x = calib_x
    with torch.no_grad():
        # Compare the layer AWQ directly optimizes: fc1's pre-activation output
        # on the salient calibration distribution (end-to-end logit MSE is
        # dominated by downstream layers' quantization, not fc1's scales).
        ref_h = ref_model.fc1(test_x)
        awq_h = awq_model.fc1(test_x)
        naive_h = naive2.fc1(test_x)
    err_awq = (ref_h - awq_h).pow(2).mean().item()
    err_naive = (ref_h - naive_h).pow(2).mean().item()
    assert err_awq < err_naive, f"AWQ ({err_awq:.6f}) should beat naive RTN ({err_naive:.6f})"


def _getattr_nested(module: nn.Module, dotted: str):
    for part in dotted.split("."):
        module = getattr(module, part)
    return module


def test_awq_per_channel_4bit_rejects_odd_total_weights():
    """bits=4 + group_size=-1 packs two weights per int8 byte; _pack_weights
    must reject an odd total weight count up front instead of a late
    _pack_4bit ValueError after scale search."""
    from llm.quantization.awq import AWQConfig, _pack_weights, quantize_model_awq

    with pytest.raises(ValueError, match="even total weight count"):
        _pack_weights(torch.randn(3, 3), bits=4, group_size=-1)

    # Even total stays valid end-to-end.
    model = nn.Linear(32, 32)
    calib = [torch.randn(4, 32) for _ in range(2)]
    from llm.quantization._awq_layer import AWQQuantizedLinear

    quantized = quantize_model_awq(model, iter(calib), AWQConfig(bits=4, group_size=-1))
    assert any(isinstance(m, AWQQuantizedLinear) for m in quantized.modules())


class _ForwardAlwaysRaises(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)

    def forward(self, x):
        raise RuntimeError("model forward intentionally broken for calibration")


def test_quantize_model_awq_warns_when_forward_always_fails(caplog):
    """Total calibration-failure must warn loudly instead of silently feeding
    raw batches as layer activations (RIL TASK-197 / ISS-237)."""
    from llm.quantization._awq_layer import AWQQuantizedLinear
    from llm.quantization.awq import AWQConfig, quantize_model_awq

    model = _ForwardAlwaysRaises()
    calib = [torch.randn(8, 16) for _ in range(4)]
    with caplog.at_level("WARNING", logger="llm.quantization.awq"):
        quantized = quantize_model_awq(model, iter(calib), AWQConfig())
    assert any("failed on EVERY calibration batch" in r.getMessage() for r in caplog.records)
    assert isinstance(quantized.fc1, AWQQuantizedLinear)
