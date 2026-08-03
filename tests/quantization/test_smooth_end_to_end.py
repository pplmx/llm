"""End-to-end tests for SmoothQuant integration on full models."""

import pytest
import torch
import torch.nn as nn


class TwoLayerMLP(nn.Module):
    """Tiny model for SmoothQuant end-to-end testing."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_quantize_model_smoothquant_replaces_all_linear_layers():
    """quantize_model_smoothquant converts every nn.Linear to SmoothQuantLinear."""
    from llm.quantization._smooth_layer import SmoothQuantLinear
    from llm.quantization.smooth import SmoothQuantConfig, quantize_model_smoothquant

    model = TwoLayerMLP(hidden=16)
    quantized = quantize_model_smoothquant(model, iter([torch.randn(8, 16)]), SmoothQuantConfig())

    linear_count = sum(1 for _ in quantized.modules() if isinstance(_, nn.Linear))
    smooth_count = sum(1 for _ in quantized.modules() if isinstance(_, SmoothQuantLinear))
    assert linear_count == 0
    assert smooth_count == 2


def test_quantize_model_smoothquant_preserves_forward_contract():
    """Quantized model accepts the same input shape and returns the same output shape."""
    from llm.quantization.smooth import SmoothQuantConfig, quantize_model_smoothquant

    torch.manual_seed(123)
    model = TwoLayerMLP(hidden=16)
    quantized = quantize_model_smoothquant(model, iter([torch.randn(8, 16)]), SmoothQuantConfig())
    out = quantized(torch.randn(2, 16))
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_quantize_model_smoothquant_preserves_bias_values():
    """Original Linear biases are copied into the SmoothQuant layers."""
    from llm.quantization.smooth import SmoothQuantConfig, quantize_model_smoothquant

    torch.manual_seed(7)
    model = TwoLayerMLP(hidden=16)
    original_biases = {
        n: m.bias.detach().clone() for n, m in model.named_modules() if n and getattr(m, "bias", None) is not None
    }
    quantized = quantize_model_smoothquant(model, iter([torch.randn(8, 16)]), SmoothQuantConfig())
    for n, m in quantized.named_modules():
        if n in original_biases and getattr(m, "bias", None) is not None:
            assert torch.allclose(m.bias.detach(), original_biases[n])


def test_quantize_model_smoothquant_rejects_already_quantized():
    """Refusing to double-quantize surfaces an actionable error."""
    from llm.quantization.smooth import SmoothQuantConfig, quantize_model_smoothquant

    model = TwoLayerMLP(hidden=16)
    quantized = quantize_model_smoothquant(model, iter([torch.randn(8, 16)]), SmoothQuantConfig())
    with pytest.raises(ValueError, match=r"already SmoothQuant-quantized"):
        quantize_model_smoothquant(quantized, iter([torch.randn(8, 16)]), SmoothQuantConfig())


def test_quantize_model_smoothquant_no_linear_raises():
    """A model without nn.Linear is rejected."""
    from llm.quantization.smooth import SmoothQuantConfig, quantize_model_smoothquant

    model = nn.Sequential(nn.GELU())
    with pytest.raises(ValueError, match=r"no nn.Linear"):
        quantize_model_smoothquant(model, iter([torch.randn(8, 16)]), SmoothQuantConfig())


def test_target_modules_filters_correctly():
    """Only the named target modules are quantized."""
    from llm.quantization._smooth_layer import SmoothQuantLinear
    from llm.quantization.smooth import SmoothQuantConfig, quantize_model_smoothquant

    model = TwoLayerMLP(hidden=16)
    quantized = quantize_model_smoothquant(
        model, iter([torch.randn(8, 16)]), SmoothQuantConfig(), target_modules=["fc1"]
    )
    assert isinstance(quantized.fc1, SmoothQuantLinear)
    assert isinstance(quantized.fc2, nn.Linear)


def test_target_modules_no_match_raises_with_available():
    """A target_modules filter matching nothing raises with available layers."""
    from llm.quantization.smooth import SmoothQuantConfig, quantize_model_smoothquant

    model = TwoLayerMLP(hidden=16)
    with pytest.raises(ValueError, match=r"matched no nn.Linear"):
        quantize_model_smoothquant(model, iter([torch.randn(8, 16)]), SmoothQuantConfig(), target_modules=["nope"])


def test_quantize_model_smoothquant_empty_calib_raises():
    """An empty calibration iterator is rejected loudly."""
    from llm.quantization.smooth import SmoothQuantConfig, quantize_model_smoothquant

    model = TwoLayerMLP(hidden=16)
    with pytest.raises(ValueError, match=r"calib_iter is empty"):
        quantize_model_smoothquant(model, iter([]), SmoothQuantConfig())


def test_quantize_model_smoothquant_falls_back_when_model_forward_fails():
    """When the model forward fails, per-layer calibration uses direct calls."""
    from llm.quantization._smooth_layer import SmoothQuantLinear
    from llm.quantization.smooth import SmoothQuantConfig, quantize_model_smoothquant

    class BrokenForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 16)

        def forward(self, x):
            raise RuntimeError("calibration forward is broken by design")

    model = BrokenForward()
    quantized = quantize_model_smoothquant(model, iter([torch.randn(8, 16)]), SmoothQuantConfig())
    assert isinstance(quantized.fc1, SmoothQuantLinear)


def test_quantize_model_smoothquant_policy_rejects_sub8bit():
    """A layer policy requesting bits=4 fails loudly for SmoothQuant."""
    from llm.quantization._policy import LayerQuantPolicy
    from llm.quantization.smooth import SmoothQuantConfig, quantize_model_smoothquant

    model = TwoLayerMLP(hidden=16)
    config = SmoothQuantConfig(layer_policies=(LayerQuantPolicy(target_modules=("fc1",), bits=4),))
    with pytest.raises(ValueError, match=r"bits must be 8"):
        quantize_model_smoothquant(model, iter([torch.randn(8, 16)]), config)


def test_quantize_model_smoothquant_policy_accepts_int8():
    """A bits=8 layer policy is a no-op and passes."""
    from llm.quantization._policy import LayerQuantPolicy
    from llm.quantization.smooth import SmoothQuantConfig, quantize_model_smoothquant

    model = TwoLayerMLP(hidden=16)
    config = SmoothQuantConfig(layer_policies=(LayerQuantPolicy(target_modules=("fc1",), bits=8),))
    quantized = quantize_model_smoothquant(model, iter([torch.randn(8, 16)]), config)
    assert quantized(torch.randn(2, 16)).shape == (2, 16)


def test_quantize_model_smoothquant_search_alpha_e2e():
    """search_alpha works at the model level."""
    from llm.quantization.smooth import SmoothQuantConfig, quantize_model_smoothquant

    torch.manual_seed(9)
    model = TwoLayerMLP(hidden=16)
    calib = torch.randn(32, 16)
    calib[:, 0] *= 20.0
    quantized = quantize_model_smoothquant(model, iter([calib]), SmoothQuantConfig(search_alpha=True))
    assert quantized(torch.randn(2, 16)).shape == (2, 16)


def test_quantize_model_smoothquant_with_collector_works():
    """quantize_model_smoothquant_with_collector materializes batches from an iterable."""
    from llm.quantization.smooth import SmoothQuantConfig, quantize_model_smoothquant_with_collector

    model = TwoLayerMLP(hidden=16)
    collector = [torch.randn(8, 16) for _ in range(5)]
    quantized = quantize_model_smoothquant_with_collector(model, collector, n_samples=3, config=SmoothQuantConfig())
    assert quantized(torch.randn(2, 16)).shape == (2, 16)


def test_quantize_model_smoothquant_with_collector_respects_n_samples():
    """Only n_samples batches are consumed from the collector."""
    from llm.quantization.smooth import SmoothQuantConfig, quantize_model_smoothquant_with_collector

    seen: list[int] = []

    def collector():
        for i in range(10):
            seen.append(i)
            yield torch.randn(8, 16)

    model = TwoLayerMLP(hidden=16)
    quantize_model_smoothquant_with_collector(model, collector(), n_samples=2, config=SmoothQuantConfig())
    assert seen == [0, 1]


def test_quantize_model_smoothquant_with_collector_rejects_non_positive_n_samples():
    """n_samples must be positive."""
    from llm.quantization.smooth import SmoothQuantConfig, quantize_model_smoothquant_with_collector

    model = TwoLayerMLP(hidden=16)
    with pytest.raises(ValueError, match=r"n_samples must be positive"):
        quantize_model_smoothquant_with_collector(model, [torch.randn(8, 16)], n_samples=0, config=SmoothQuantConfig())


def test_quantize_model_smoothquant_reduces_error_on_outlier_model():
    """On a model with activation outliers, SmoothQuant beats no smoothing end-to-end."""
    from llm.quantization._smooth_layer import SmoothQuantLinear
    from llm.quantization.smooth import (
        SmoothQuantConfig,
        quantize_model_smoothquant,
    )

    torch.manual_seed(11)
    calib_x = torch.randn(64, 32)
    calib_x[:, 0] *= 30.0  # outlier channel into fc1
    calib = [calib_x]

    base = TwoLayerMLP(hidden=32)
    ref_model = __import__("copy").deepcopy(base)
    smooth_model = quantize_model_smoothquant(base, iter(calib), SmoothQuantConfig(alpha=0.5))

    # Naive baseline: same int8 storage but input_scales=1 (no smoothing) and
    # act_scale from the raw activations.
    naive = __import__("copy").deepcopy(ref_model)
    with torch.no_grad():
        for name, mod in naive.named_modules():
            if isinstance(mod, nn.Linear):
                act_max = calib_x.abs().max(dim=0)[0].float()
                w = mod.weight.data.float()
                weight_scales = (w.abs().max(dim=1, keepdim=True)[0] / 127.0).clamp(min=1e-8)
                weight_packed = torch.round(w / weight_scales).clamp(-128, 127).to(torch.int8).flatten()
                act_scale = torch.tensor(act_max.max().item() / 127.0)
                replacement = SmoothQuantLinear(
                    in_features=mod.in_features,
                    out_features=mod.out_features,
                    bias=mod.bias is not None,
                    weight_packed=weight_packed,
                    weight_scales=weight_scales.to(torch.float16),
                    act_scale=act_scale.to(torch.float16),
                    sym=True,
                    input_scales=None,
                )
                if mod.bias is not None:
                    replacement.bias.copy_(mod.bias.data)
                parent_name, _, child_name = name.rpartition(".")
                parent = naive if not parent_name else _getattr_nested(naive, parent_name)
                setattr(parent, child_name, replacement)

    test_x = calib_x
    with torch.no_grad():
        ref_h = ref_model.fc1(test_x)
        smooth_h = smooth_model.fc1(test_x)
        naive_h = naive.fc1(test_x)
    err_smooth = (ref_h - smooth_h).pow(2).mean().item()
    err_naive = (ref_h - naive_h).pow(2).mean().item()
    assert err_smooth < err_naive, f"SmoothQuant ({err_smooth:.6f}) should beat no smoothing ({err_naive:.6f})"


def _getattr_nested(module: nn.Module, dotted: str):
    for part in dotted.split("."):
        module = getattr(module, part)
    return module
