"""Q4_0 / Q8_0 block quantization: layout, reference vectors, error bounds."""

from __future__ import annotations

import numpy as np
import pytest

from llm.export.gguf.quant import (
    dequantize_q4_0,
    dequantize_q8_0,
    quantize_q4_0,
    quantize_q8_0,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


class TestQ40:
    """GGML_TYPE_Q4_0: fp16 scale + 2-per-byte nibbles, offset 8."""

    def test_zero_input_packs_eight_nibble(self):
        packed, scales = quantize_q4_0(np.zeros(64, dtype=np.float32))
        # (int8_t)(0 + 8.5) truncates to 8 → every byte 0x88.
        assert np.array_equal(packed, np.full(32, 0x88, dtype=np.uint8))
        assert np.array_equal(scales, np.zeros(2, dtype=np.float16))
        assert np.array_equal(dequantize_q4_0(packed, scales, 64), np.zeros(64))

    def test_nibble_layout_low_first_half_high_second_half(self):
        # One block: low nibbles feed elements 0..15, high nibbles feed elements 16..31.
        packed = np.zeros(16, dtype=np.uint8)
        packed[0] = 0x12  # low 2 → element 0, high 1 → element 16
        packed[1] = 0x34  # low 4 → element 1, high 3 → element 17
        out = dequantize_q4_0(packed, np.array([1.0], dtype=np.float16), 32)
        assert out[0] == 2 - 8
        assert out[16] == 1 - 8
        assert out[1] == 4 - 8
        assert out[17] == 3 - 8
        assert out[31] == 0 - 8

    def test_reference_vector_matches_ggml_math(self):
        # amax = 7 → d = 1.0; nibble = trunc(x + 8.5): -7 → 1, 7 → 15.
        x = np.array([-7.0, 7.0] * 16, dtype=np.float32)
        packed, scales = quantize_q4_0(x)
        assert np.array_equal(scales, np.array([1.0], dtype=np.float16))
        assert np.array_equal(packed, np.array([0x11, 0xFF] * 8, dtype=np.uint8))
        assert np.allclose(dequantize_q4_0(packed, scales, 32), x, atol=0.0)

    def test_roundtrip_error_is_bounded_by_block_scale(self, rng):
        x = rng.normal(size=256).astype(np.float32)
        packed, scales = quantize_q4_0(x)
        out = dequantize_q4_0(packed, scales, x.size)
        assert np.max(np.abs(out - x)) <= float(np.max(scales)) + 1e-6

    def test_requires_multiple_of_block_size(self):
        with pytest.raises(ValueError, match="multiple of 32"):
            quantize_q4_0(np.zeros(10, dtype=np.float32))

    def test_scale_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="expected 2 Q4_0 scales"):
            dequantize_q4_0(np.zeros(32, dtype=np.uint8), np.array([1.0]), 64)


class TestQ80:
    """GGML_TYPE_Q8_0: fp16 scale + 32 int8 values."""

    def test_zero_input(self):
        values, scales = quantize_q8_0(np.zeros(64, dtype=np.float32))
        assert np.array_equal(values, np.zeros(64, dtype=np.int8))
        assert np.array_equal(scales, np.zeros(2, dtype=np.float16))
        assert np.array_equal(dequantize_q8_0(values, scales, 64), np.zeros(64))

    def test_constant_vector_uses_full_int8_range(self):
        x = np.full(32, 3.0, dtype=np.float32)
        values, scales = quantize_q8_0(x)
        assert np.array_equal(scales, np.array([3.0 / 127.0], dtype=np.float16))
        assert np.array_equal(values, np.full(32, 127, dtype=np.int8))
        assert np.allclose(dequantize_q8_0(values, scales, 32), x, atol=3.0 / 127.0)

    def test_values_map_directly(self):
        values = np.zeros(32, dtype=np.int8)
        values[0] = 1
        values[1] = -2
        values[2] = 3
        out = dequantize_q8_0(values, np.array([2.0]), 32)
        assert out[0] == 2.0
        assert out[1] == -4.0
        assert out[2] == 6.0

    def test_roundtrip_error_is_bounded(self, rng):
        x = rng.normal(size=256).astype(np.float32)
        values, scales = quantize_q8_0(x)
        out = dequantize_q8_0(values, scales, x.size)
        assert np.max(np.abs(out - x)) <= float(np.max(scales)) / 2.0 + 1e-6

    def test_requires_multiple_of_block_size(self):
        with pytest.raises(ValueError, match="multiple of 32"):
            quantize_q8_0(np.zeros(10, dtype=np.float32))

    def test_scale_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="expected 2 Q8_0 scales"):
            dequantize_q8_0(np.zeros(64, dtype=np.int8), np.array([1.0]), 64)
