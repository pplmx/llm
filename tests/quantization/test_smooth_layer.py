"""Tests for SmoothQuantLinear storage layer (int8 weights + activation quant)."""

import pytest
import torch


def _build_smooth_layer(*, in_features: int = 16, out_features: int = 8, bias: bool = True):
    """Build a valid SmoothQuantLinear with deterministic contents."""
    from llm.quantization.smooth import _quantize_layer_components

    torch.manual_seed(0)
    w = torch.randn(out_features, in_features)
    act_max = torch.rand(in_features) + 0.1
    weight_packed, weight_scales, act_scale, input_scales = _quantize_layer_components(w, act_max, alpha=0.5)
    return __import__("llm.quantization._smooth_layer", fromlist=["SmoothQuantLinear"]).SmoothQuantLinear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        weight_packed=weight_packed,
        weight_scales=weight_scales.to(torch.float16),
        act_scale=act_scale.to(torch.float16),
        sym=True,
        input_scales=input_scales.to(torch.float16),
    )


def test_smooth_layer_initializes_buffers():
    """SmoothQuantLinear exposes int8 weights, weight/act scales, input scales."""
    from llm.quantization._smooth_layer import SmoothQuantLinear

    layer = _build_smooth_layer()
    assert isinstance(layer, SmoothQuantLinear)
    assert layer.weight_packed.dtype == torch.int8
    assert layer.weight_packed.numel() == 8 * 16
    assert layer.weight_scales.shape == (8, 1)
    assert layer.input_scales.shape == (16,)
    assert layer.bias is not None


def test_smooth_layer_forward_contract():
    """forward accepts [..., in_features] and returns [..., out_features]."""
    layer = _build_smooth_layer()
    out = layer(torch.randn(2, 3, 16))
    assert out.shape == (2, 3, 8)
    assert torch.isfinite(out).all()


def test_smooth_layer_forward_matches_manual_math():
    """forward equals the exact fake-quantized W+A computation."""
    torch.manual_seed(1)
    layer = _build_smooth_layer()
    x = torch.randn(3, 16)

    w_fp = layer._dequantize_weights()
    x_s = x / layer.input_scales.float()
    act_scale = layer.act_scale.float()
    x_q = torch.clamp(torch.round(x_s / act_scale), -128, 127) * act_scale
    expected = torch.nn.functional.linear(x_q, w_fp, layer.bias)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_smooth_layer_activation_fake_quant_is_int8():
    """The forward quantizes activations to INT8 levels (at most 256 distinct)."""
    layer = _build_smooth_layer()
    x = torch.randn(64, 16) * 5.0
    x_s = x / layer.input_scales.float()
    act_scale = layer.act_scale.float()
    x_q = torch.clamp(torch.round(x_s / act_scale), -128, 127)
    assert x_q.unique().numel() <= 256


def test_smooth_layer_rejects_asymmetric_forward():
    """sym=False is not implemented at forward time."""
    from llm.quantization._smooth_layer import SmoothQuantLinear
    from llm.quantization.smooth import _quantize_layer_components

    torch.manual_seed(0)
    w = torch.randn(8, 16)
    act_max = torch.rand(16) + 0.1
    weight_packed, weight_scales, act_scale, input_scales = _quantize_layer_components(w, act_max, alpha=0.5)
    layer = SmoothQuantLinear(
        in_features=16,
        out_features=8,
        bias=False,
        weight_packed=weight_packed,
        weight_scales=weight_scales.to(torch.float16),
        act_scale=act_scale.to(torch.float16),
        sym=False,
        input_scales=input_scales.to(torch.float16),
    )
    with pytest.raises(NotImplementedError, match="Asymmetric"):
        layer(torch.randn(2, 16))


def test_smooth_layer_bias_preserved():
    """The bias parameter is present and trainable when the source had bias."""
    layer = _build_smooth_layer(bias=True)
    assert layer.bias is not None
    assert layer.bias.requires_grad


def test_smooth_layer_no_input_scales_path():
    """input_scales=None skips the smoothing division."""
    layer = _build_smooth_layer()
    layer.input_scales = None
    out = layer(torch.randn(2, 16))
    assert out.shape == (2, 8)


def test_smooth_layer_forward_accepts_fp16_bf16_input():
    """fp16/bf16 inputs must not crash (regression for ISS-018).

    RIL ISS-191: the output follows the input dtype (native Linear
    semantics) so the quantized layer feeds surrounding half-precision
    linears without a dtype crash; it no longer always emits fp32."""
    layer = _build_smooth_layer()
    x32 = torch.randn(2, 16)
    ref = layer(x32)
    for x in (x32.half(), x32.bfloat16()):
        out = layer(x)
        assert out.dtype == torch.float32
        assert torch.allclose(out, ref, atol=5e-2)
