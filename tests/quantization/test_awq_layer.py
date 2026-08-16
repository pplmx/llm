"""Tests for AWQQuantizedLinear storage layer (packed storage + input scales)."""

import pytest
import torch


def _build_awq_layer(
    *,
    in_features: int = 16,
    out_features: int = 8,
    bits: int = 4,
    group_size: int = 8,
    input_scales: torch.Tensor | None = None,
    bias: bool = True,
):
    """Build a valid AWQQuantizedLinear with deterministic contents."""
    from llm.quantization.awq import _pack_weights

    torch.manual_seed(0)
    w = torch.randn(out_features, in_features)
    packed, scales, effective_gs = _pack_weights(w, bits, group_size)
    return __import__("llm.quantization._awq_layer", fromlist=["AWQQuantizedLinear"]).AWQQuantizedLinear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        weight_packed=packed,
        scales=scales.to(torch.float16),
        bits=bits,
        group_size=effective_gs,
        sym=True,
        input_scales=input_scales,
    )


def test_awq_layer_initializes_buffers():
    """AWQQuantizedLinear exposes packed weights, scales, and input_scales."""
    from llm.quantization._awq_layer import AWQQuantizedLinear

    input_scales = torch.ones(16, dtype=torch.float16)
    layer = _build_awq_layer(input_scales=input_scales)
    assert isinstance(layer, AWQQuantizedLinear)
    assert layer.weight_packed.shape == (8 * 16 // 2,)
    assert layer.scales.dtype == torch.float16
    assert layer.input_scales is not None
    assert layer.bias is not None


def test_awq_layer_no_input_scales_path():
    """input_scales=None leaves the forward without the x/s compensation."""
    layer = _build_awq_layer(input_scales=None)
    assert layer.input_scales is None
    x = torch.randn(2, 16)
    out = layer(x)
    assert out.shape == (2, 8)


def test_awq_layer_forward_applies_input_scale_compensation():
    """forward computes Q(W·s)·(x/s): scaling x by k must invert the output."""
    torch.manual_seed(0)
    input_scales = torch.rand(16, dtype=torch.float16) + 0.5
    layer = _build_awq_layer(input_scales=input_scales)

    x = torch.randn(3, 16)
    out = layer(x)
    out_scaled = layer(x * 2.0)
    # The input-scale division is linear in x (Q is fixed), so doubling the
    # input doubles the output — up to fp16 rounding.
    assert torch.allclose(out_scaled, out * 2.0, atol=1e-3)


def test_awq_layer_forward_matches_manual_dequant():
    """forward equals F.linear(x/s, dequantized W·s, bias)."""
    torch.manual_seed(1)
    input_scales = torch.rand(16, dtype=torch.float16) + 0.5
    layer = _build_awq_layer(input_scales=input_scales)
    x = torch.randn(2, 16)

    # Manual dequantization matching _pack_weights math.
    unpacked = layer._unpack_weights().to(torch.float32)
    signed = unpacked - 8.0 if layer.bits == 4 else unpacked
    gs = layer.group_size
    if gs == -1:
        scales_expanded = layer.scales.to(torch.float32)
    else:
        scales_expanded = layer.scales.to(torch.float32).repeat_interleave(gs, dim=1)
    w_fp = signed * scales_expanded
    expected = torch.nn.functional.linear(x / input_scales.float(), w_fp, layer.bias)
    assert torch.allclose(layer(x), expected, atol=1e-4)


def test_awq_layer_4bit_packed_storage_is_half_size():
    """4-bit packed storage halves the byte count vs unpacked int4."""
    layer = _build_awq_layer(bits=4)
    numel = layer.out_features * layer.in_features
    assert layer.weight_packed.numel() == numel // 2


def test_awq_layer_8bit_storage():
    """8-bit storage keeps one int8 per weight (no packing)."""
    layer = _build_awq_layer(bits=8)
    numel = layer.out_features * layer.in_features
    assert layer.weight_packed.numel() == numel


def test_awq_layer_rejects_asymmetric_forward():
    """sym=False is not implemented at forward time."""
    from llm.quantization._awq_layer import AWQQuantizedLinear
    from llm.quantization.awq import _pack_weights

    torch.manual_seed(0)
    w = torch.randn(8, 16)
    packed, scales, gs = _pack_weights(w, 4, 8)
    layer = AWQQuantizedLinear(
        in_features=16,
        out_features=8,
        bias=False,
        weight_packed=packed,
        scales=scales.to(torch.float16),
        bits=4,
        group_size=gs,
        sym=False,
    )
    with pytest.raises(NotImplementedError, match="Asymmetric"):
        layer(torch.randn(2, 16))


def test_awq_layer_per_channel_group_mode():
    """group_size=-1 (per-channel) dequantizes with a [out_f, 1] scale."""
    from llm.quantization.awq import _pack_weights

    torch.manual_seed(2)
    w = torch.randn(8, 16)
    _packed, scales, gs = _pack_weights(w, 4, -1)
    assert gs == -1
    assert scales.shape == (8, 1)
    layer = _build_awq_layer(group_size=-1)
    assert layer(torch.randn(3, 16)).shape == (3, 8)


def test_awq_layer_bias_preserved():
    """The bias parameter is present and trainable when the source had bias."""
    layer = _build_awq_layer(bias=True)
    assert layer.bias is not None
    assert layer.bias.requires_grad


def test_awq_layer_forward_accepts_fp16_bf16_input():
    """fp16/bf16 inputs must not crash (regression for ISS-018).

    RIL ISS-191: the output follows the input dtype (native Linear
    semantics) so the quantized layer feeds surrounding half-precision
    linears without a dtype crash; it no longer always emits fp32."""
    layer = _build_awq_layer()
    x32 = torch.randn(3, 16)
    ref = layer(x32)
    for x in (x32.half(), x32.bfloat16()):
        out = layer(x)
        assert out.dtype == torch.float32
        assert torch.allclose(out, ref, atol=1e-2)
