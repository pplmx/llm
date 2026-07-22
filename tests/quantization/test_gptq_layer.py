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
