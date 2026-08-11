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
        # ggml quantize_row_q4_0_ref: max = *first* value reaching the block's
        # max |x| = -7 (index 0 wins the tie over +7), d = max/-8 = -7/-8 =
        # +0.875, id = 8/7, q = MIN(15, trunc(x*id + 8.5)).
        #   x=-7 -> trunc(-8 + 8.5) = trunc(0.5)  = 0
        #   x=+7 -> trunc( 8 + 8.5) = trunc(16.5) = 16 -> clipped 15
        # x = [-7,+7,-7,+7,...] => even elements (incl. element 0, the low
        # nibble) are -7 -> 0, odd elements (incl. element 16, the high
        # nibble) are +7 -> 15.  byte = low | high<<4, so bytes alternate
        # 0x00 (element 0/16 both -7) and 0xFF (element 1/17 both +7).
        x = np.array([-7.0, 7.0] * 16, dtype=np.float32)
        packed, scales = quantize_q4_0(x)
        assert np.array_equal(scales, np.array([0.875], dtype=np.float16))
        assert np.array_equal(packed, np.array([0x00, 0xFF] * 8, dtype=np.uint8))
        assert np.allclose(dequantize_q4_0(packed, scales, 32), x, atol=abs(scales[0]) + 1e-6)

    def test_reference_matches_ggml_oracle(self):
        """Byte-exact match against ggml's quantize_row_q4_0_reference.

        The module's earlier 'amax/7 + trunc(x*7/amax+8.5)' formula was
        self-consistent (roundtriped through its own dequant) but produced
        bytes no llama.cpp build reads back: ggml stores the *signed* extreme
        over -8 and packs low/high nibbles the same way but with that scale.
        Pin the exact ggml convention so exported Q4_0 tensors are actually
        consumable by the ecosystem.
        """
        for seed in range(8):
            x = np.random.default_rng(seed).normal(size=128).astype(np.float32)
            packed, scales = quantize_q4_0(x)
            # Independent transcription of ggml-quants.c.
            blocks = x.reshape(-1, 32)
            gold_packed = np.zeros((blocks.shape[0], 16), dtype=np.uint8)
            gold_scales = np.zeros(blocks.shape[0], dtype=np.float16)
            for b in range(blocks.shape[0]):
                amax, mx = 0.0, 0.0
                for v in blocks[b]:
                    if amax < abs(float(v)):
                        amax = abs(float(v))
                        mx = float(v)
                d = mx / -8.0
                ident = 1.0 / d if d != 0 else 0.0
                gold_scales[b] = d
                for j in range(16):
                    xi0 = min(15, int(np.trunc(float(blocks[b][0 + j]) * ident + 8.5)))
                    xi1 = min(15, int(np.trunc(float(blocks[b][16 + j]) * ident + 8.5)))
                    gold_packed[b][j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4)
            assert np.array_equal(packed, gold_packed.reshape(-1)), f"seed {seed} packed"
            assert np.array_equal(scales, gold_scales), f"seed {seed} scales"

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
