"""Tests for GPTQQuantizedLinear storage layer (packed 4-bit, dequantize, forward)."""

import pytest
import torch

# === Pack/unpack correctness ===


def test_pack_4bit_round_trip():
    """Pack unsigned int4 values into int8 storage and unpack back."""
    from llm.quantization._gptq_layer import _pack_4bit, _unpack_4bit

    # Values in [0, 15]
    w = torch.tensor([0, 1, 7, 8, 15, 5, 9, 3], dtype=torch.int8)
    packed = _pack_4bit(w)
    unpacked = _unpack_4bit(packed, numel=len(w))
    assert torch.equal(unpacked, w)


def test_pack_4bit_even_length_required():
    """_pack_4bit requires even number of values (pairs into single int8)."""
    from llm.quantization._gptq_layer import _pack_4bit

    with pytest.raises(ValueError, match="even"):
        _pack_4bit(torch.tensor([1, 2, 3], dtype=torch.int8))  # 3 elements


def test_pack_4bit_rejects_out_of_range():
    """Values must be in [0, 15] (unsigned int4)."""
    from llm.quantization._gptq_layer import _pack_4bit

    with pytest.raises(ValueError, match="range"):
        _pack_4bit(torch.tensor([1, 16], dtype=torch.int8))  # 16 is out of range


def test_packed_storage_is_half_size():
    """4-bit packed: stored bytes == numel / 2. This is the 4-bit promise."""
    from llm.quantization._gptq_layer import _pack_4bit

    w = torch.randint(0, 16, (4, 32), dtype=torch.int8)  # 128 values
    packed = _pack_4bit(w)
    assert packed.numel() == 64  # 128 / 2
    assert packed.element_size() == 1  # int8 storage
    # Total bytes: 64, vs 128 for unpacked int8
    assert packed.untyped_storage().nbytes() == 64


# === GPTQQuantizedLinear storage ===


def test_gptq_layer_initializes_with_correct_buffers():
    """GPTQQuantizedLinear exposes packed weight, scales, zeros buffers."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear

    layer = GPTQQuantizedLinear(
        in_features=16,
        out_features=8,
        bias=True,
        weight_packed=torch.zeros(8 * 16 // 2, dtype=torch.int8),
        scales=torch.ones(8, 2, dtype=torch.float16),  # 16/8 = 2 groups
        zeros=None,
        bits=4,
        group_size=8,
        sym=True,
    )
    assert layer.in_features == 16
    assert layer.out_features == 8
    assert layer.weight_packed.shape == (64,)
    assert layer.scales.shape == (8, 2)
    assert layer.bias is not None


def test_gptq_layer_no_bias_path():
    """bias=False creates a layer without bias parameter."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear

    layer = GPTQQuantizedLinear(
        in_features=8,
        out_features=4,
        bias=False,
        weight_packed=torch.zeros(16, dtype=torch.int8),
        scales=torch.ones(4, 1, dtype=torch.float16),
        zeros=None,
        bits=4,
        group_size=-1,
        sym=True,
    )
    assert layer.bias is None


def test_gptq_layer_unpack_matches_original():
    """Internal _unpack_weights returns int8 values that match original quantization."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear, _pack_4bit

    # Create known quantized weights
    original_int4 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int8)
    packed = _pack_4bit(original_int4.flatten()).reshape(2, 2)

    layer = GPTQQuantizedLinear(
        in_features=4,
        out_features=2,
        bias=False,
        weight_packed=packed,
        scales=torch.ones(2, 1, dtype=torch.float16),
        zeros=None,
        bits=4,
        group_size=-1,
        sym=True,
    )
    unpacked = layer._unpack_weights()
    assert torch.equal(unpacked, original_int4)


# === Forward correctness ===


def test_forward_close_to_fp32_baseline():
    """Forward output cosine_sim > 0.99 vs equivalent fp32 Linear."""
    from torch import nn

    from llm.quantization._gptq_layer import GPTQQuantizedLinear, _pack_4bit

    torch.manual_seed(99)
    in_f, out_f = 32, 16

    # Original fp32 weight (small magnitude for tight 4-bit fit)
    w_fp32 = torch.randn(out_f, in_f) * 0.1
    x = torch.randn(8, in_f)

    # Quantize via simple per-channel 4-bit
    abs_max = w_fp32.abs().max(dim=1, keepdim=True)[0]
    qmax = 7  # 2^(4-1) - 1
    scale = (abs_max / qmax).clamp(min=1e-8)
    w_int4 = torch.round(w_fp32 / scale).clamp(-8, 7).to(torch.int8) + 8  # shift to [0, 15]
    packed = _pack_4bit(w_int4.flatten())

    layer_q = GPTQQuantizedLinear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        weight_packed=packed,
        scales=scale.to(torch.float16),
        zeros=None,
        bits=4,
        group_size=-1,
        sym=True,
    )

    # fp32 baseline
    layer_fp32 = nn.Linear(in_f, out_f, bias=False)
    with torch.no_grad():
        layer_fp32.weight.copy_(w_fp32)

    out_q = layer_q(x)
    out_fp = layer_fp32(x)

    cosine = torch.nn.functional.cosine_similarity(out_q, out_fp, dim=-1).mean()
    mse = ((out_q - out_fp) ** 2).mean().item()

    assert cosine > 0.99, f"cosine_sim {cosine:.4f} too low"
    assert mse < 1e-2


def test_forward_preserves_bias():
    """Bias is added correctly to forward output."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear

    in_f, out_f = 8, 4
    bias_values = torch.tensor([0.1, -0.2, 0.3, -0.4])

    layer = GPTQQuantizedLinear(
        in_features=in_f,
        out_features=out_f,
        bias=True,
        weight_packed=torch.zeros(out_f * in_f // 2, dtype=torch.int8),
        scales=torch.ones(out_f, 1, dtype=torch.float16) * 0.01,  # tiny scale → near-zero weight
        zeros=None,
        bits=4,
        group_size=-1,
        sym=True,
    )
    with torch.no_grad():
        layer.bias.copy_(bias_values)

    x = torch.zeros(2, in_f)
    out = layer(x)

    # Weights are ~0, so output ≈ bias
    assert torch.allclose(out, bias_values.unsqueeze(0).expand(2, -1), atol=1e-3)


def test_forward_grouped_quantization():
    """group_size=8 with per-group scales produces correct output."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear, _pack_4bit

    in_f, out_f = 16, 4
    group_size = 8

    # Per-group distinct scales
    scales = torch.tensor([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.4, 0.8]], dtype=torch.float16)

    # Pack unsigned=8 → signed=0, so weights are all 0 after dequantize
    w_int4 = torch.full((out_f * in_f,), 8, dtype=torch.int8)
    packed = _pack_4bit(w_int4)

    layer = GPTQQuantizedLinear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        weight_packed=packed,
        scales=scales,
        zeros=None,
        bits=4,
        group_size=group_size,
        sym=True,
    )

    x = torch.ones(1, in_f)
    out = layer(x)
    # All weights are 0, output is 0
    assert torch.allclose(out, torch.zeros(1, out_f), atol=1e-3)


def test_forward_input_shape_2d_and_3d():
    """Layer accepts both [batch, in] and [batch, seq, in]."""
    from llm.quantization._gptq_layer import GPTQQuantizedLinear

    in_f, out_f = 8, 4
    layer = GPTQQuantizedLinear(
        in_features=in_f,
        out_features=out_f,
        bias=False,
        weight_packed=torch.zeros(out_f * in_f // 2, dtype=torch.int8),
        scales=torch.ones(out_f, 1, dtype=torch.float16) * 0.01,
        zeros=None,
        bits=4,
        group_size=-1,
        sym=True,
    )

    x_2d = torch.randn(3, in_f)
    x_3d = torch.randn(2, 5, in_f)

    out_2d = layer(x_2d)
    out_3d = layer(x_3d)

    assert out_2d.shape == (3, out_f)
    assert out_3d.shape == (2, 5, out_f)
