"""Q4_0 / Q8_0 block quantization: layout, reference vectors, error bounds."""

from __future__ import annotations

import numpy as np
import pytest

from llm.export.gguf.quant import (
    dequantize_q2_k,
    dequantize_q4_0,
    dequantize_q4_k,
    dequantize_q5_k,
    dequantize_q6_k,
    dequantize_q8_0,
    quantize_q2_k,
    quantize_q4_0,
    quantize_q4_k,
    quantize_q5_k,
    quantize_q6_k,
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


Q6_K_BLOCK_BYTES = 210  # ql(128) + qh(64) + scales(16) + d(2)
Q6_K_BLOCK = 256


class TestQ6K:
    """Q6_K (256-wide) write path: layout, zero handling, error bounds."""

    def test_block_byte_size(self, rng):
        raw = quantize_q6_k(rng.normal(size=1024).astype(np.float32))  # 4 blocks
        assert len(raw) == 4 * Q6_K_BLOCK_BYTES

    def test_zero_block_decodes_to_zero(self):
        raw = quantize_q6_k(np.zeros(Q6_K_BLOCK, dtype=np.float32))
        assert len(raw) == Q6_K_BLOCK_BYTES
        out = dequantize_q6_k(raw, Q6_K_BLOCK)
        assert np.all(out == 0.0)

    def test_constant_block_reconstructs_consistently(self):
        c = 3.5
        x = np.full(Q6_K_BLOCK, c, dtype=np.float32)
        out = dequantize_q6_k(quantize_q6_k(x), x.size)
        # every element shares the same (d, sc, q), so reconstruction is a
        # single consistent value with small relative error (6-bit grid).
        assert np.all(out == out[0])
        assert abs(out[0] - c) / abs(c) < 0.05

    def test_roundtrip_error_is_bounded(self, rng):
        x = rng.standard_normal(Q6_K_BLOCK * 3).astype(np.float32) * 2.0
        out = dequantize_q6_k(quantize_q6_k(x), x.size)
        rel = np.mean(np.abs(out - x)) / (np.mean(np.abs(x)) + 1e-9)
        assert rel < 0.04, f"Q6_K relative error too large: {rel:.4f}"

    def test_mixed_magnitude_blocks_roundtrip(self, rng):
        x = np.concatenate(
            [
                rng.standard_normal(Q6_K_BLOCK) * 100.0,
                rng.standard_normal(Q6_K_BLOCK) * 0.001,
                rng.standard_normal(Q6_K_BLOCK) * 5.0,
            ]
        ).astype(np.float32)
        out = dequantize_q6_k(quantize_q6_k(x), x.size)
        assert np.all(np.isfinite(out))
        rel = np.mean(np.abs(out - x)) / (np.mean(np.abs(x)) + 1e-12)
        assert rel < 0.5

    def test_requires_multiple_of_256(self):
        with pytest.raises(ValueError, match="multiple of 256"):
            quantize_q6_k(np.zeros(100, dtype=np.float32))

    def test_exporter_resolves_and_picks_q6k(self):
        from llm.export.gguf.exporter import _pick_tensor_type, _resolve_quant_type
        from llm.export.gguf.spec import GGMLQuantizationType

        assert _resolve_quant_type("q6_k") is GGMLQuantizationType.Q6_K
        el = GGMLQuantizationType.Q6_K
        assert _pick_tensor_type(np.zeros((4, 256), np.float32), el, 2) is el
        # last dim not a multiple of 256 -> falls back to F16.
        assert _pick_tensor_type(np.zeros((4, 100), np.float32), el, 2) is GGMLQuantizationType.F16

    def test_writer_reader_roundtrip(self, tmp_path, rng):
        """Writer emits Q6_K; GGUFReader dequantizes it back close to source."""
        from llm.export.gguf import GGUFReader, GGUFWriter

        x = rng.standard_normal(Q6_K_BLOCK * 2).astype(np.float32)
        path = tmp_path / "q6k.gguf"
        writer = GGUFWriter(path)
        writer.add_tensor("w", x, ggml_type="q6_k")
        writer.write()
        reader = GGUFReader(path)
        out = reader.read_tensor("w")
        assert out.shape == x.shape
        rel = np.mean(np.abs(out - x)) / (np.mean(np.abs(x)) + 1e-9)
        assert rel < 0.04


Q4_K_BLOCK_BYTES = 144  # d(2) + dmin(2) + scales(12) + qs(128)


class TestQ4K:
    """Q4_K (256-wide) write path: layout, zero handling, error bounds."""

    def test_block_byte_size(self, rng):
        raw = quantize_q4_k(rng.normal(size=1024).astype(np.float32))  # 4 blocks
        assert len(raw) == 4 * Q4_K_BLOCK_BYTES

    def test_zero_block_decodes_to_zero(self):
        raw = quantize_q4_k(np.zeros(Q6_K_BLOCK, dtype=np.float32))
        assert len(raw) == Q4_K_BLOCK_BYTES
        assert np.all(dequantize_q4_k(raw, Q6_K_BLOCK) == 0.0)

    def test_roundtrip_error_is_bounded(self, rng):
        x = rng.standard_normal(Q6_K_BLOCK * 3).astype(np.float32) * 2.0
        out = dequantize_q4_k(quantize_q4_k(x), x.size)
        rel = np.mean(np.abs(out - x)) / (np.mean(np.abs(x)) + 1e-9)
        # Q4_K is 4-bit so it is coarser than Q6_K; keep a generous bound that
        # still catches a wrong layout (which would blow the error up).
        assert rel < 0.12, f"Q4_K relative error too large: {rel:.4f}"

    def test_centered_data_roundtrip(self, rng):
        x = rng.standard_normal(Q6_K_BLOCK * 2).astype(np.float32)
        out = dequantize_q4_k(quantize_q4_k(x), x.size)
        assert np.all(np.isfinite(out))
        rel = np.mean(np.abs(out - x)) / (np.mean(np.abs(x)) + 1e-9)
        assert rel < 0.1

    def test_requires_multiple_of_256(self):
        with pytest.raises(ValueError, match="multiple of 256"):
            quantize_q4_k(np.zeros(100, dtype=np.float32))

    def test_writer_reader_roundtrip(self, tmp_path, rng):
        """Writer emits Q4_K; GGUFReader dequantizes it back close to source."""
        from llm.export.gguf import GGUFReader, GGUFWriter

        x = rng.standard_normal(Q6_K_BLOCK * 2).astype(np.float32)
        path = tmp_path / "q4k.gguf"
        writer = GGUFWriter(path)
        writer.add_tensor("w", x, ggml_type="q4_k")
        writer.write()
        reader = GGUFReader(path)
        out = reader.read_tensor("w")
        assert out.shape == x.shape
        rel = np.mean(np.abs(out - x)) / (np.mean(np.abs(x)) + 1e-9)
        assert rel < 0.12

    def test_exporter_resolves_and_picks_q4k(self):
        from llm.export.gguf.exporter import _pick_tensor_type, _resolve_quant_type
        from llm.export.gguf.spec import GGMLQuantizationType

        assert _resolve_quant_type("q4_k") is GGMLQuantizationType.Q4_K
        el = GGMLQuantizationType.Q4_K
        assert _pick_tensor_type(np.zeros((4, 256), np.float32), el, 2) is el
        assert _pick_tensor_type(np.zeros((4, 100), np.float32), el, 2) is GGMLQuantizationType.F16


Q5_K_BLOCK_BYTES = 176  # d(2) + dmin(2) + scales(12) + qh(32) + qs(128)


class TestQ5K:
    """Q5_K (256-wide) write path: layout, zero handling, error bounds."""

    def test_block_byte_size(self, rng):
        raw = quantize_q5_k(rng.normal(size=1024).astype(np.float32))  # 4 blocks
        assert len(raw) == 4 * Q5_K_BLOCK_BYTES

    def test_zero_block_decodes_to_zero(self):
        raw = quantize_q5_k(np.zeros(Q6_K_BLOCK, dtype=np.float32))
        assert len(raw) == Q5_K_BLOCK_BYTES
        assert np.all(dequantize_q5_k(raw, Q6_K_BLOCK) == 0.0)

    def test_roundtrip_error_is_bounded(self, rng):
        x = rng.standard_normal(Q6_K_BLOCK * 3).astype(np.float32) * 2.0
        out = dequantize_q5_k(quantize_q5_k(x), x.size)
        rel = np.mean(np.abs(out - x)) / (np.mean(np.abs(x)) + 1e-9)
        # 5-bit sits between Q4_K and Q6_K; a wrong layout blows the error up.
        assert rel < 0.07, f"Q5_K relative error too large: {rel:.4f}"

    def test_centered_data_roundtrip(self, rng):
        x = rng.standard_normal(Q6_K_BLOCK * 2).astype(np.float32)
        out = dequantize_q5_k(quantize_q5_k(x), x.size)
        assert np.all(np.isfinite(out))
        rel = np.mean(np.abs(out - x)) / (np.mean(np.abs(x)) + 1e-9)
        assert rel < 0.08

    def test_requires_multiple_of_256(self):
        with pytest.raises(ValueError, match="multiple of 256"):
            quantize_q5_k(np.zeros(100, dtype=np.float32))

    def test_writer_reader_roundtrip(self, tmp_path, rng):
        """Writer emits Q5_K; GGUFReader dequantizes it back close to source."""
        from llm.export.gguf import GGUFReader, GGUFWriter

        x = rng.standard_normal(Q6_K_BLOCK * 2).astype(np.float32)
        path = tmp_path / "q5k.gguf"
        writer = GGUFWriter(path)
        writer.add_tensor("w", x, ggml_type="q5_k")
        writer.write()
        reader = GGUFReader(path)
        out = reader.read_tensor("w")
        assert out.shape == x.shape
        rel = np.mean(np.abs(out - x)) / (np.mean(np.abs(x)) + 1e-9)
        assert rel < 0.08

    def test_exporter_resolves_and_picks_q5k(self):
        from llm.export.gguf.exporter import _pick_tensor_type, _resolve_quant_type
        from llm.export.gguf.spec import GGMLQuantizationType

        assert _resolve_quant_type("q5_k") is GGMLQuantizationType.Q5_K
        el = GGMLQuantizationType.Q5_K
        assert _pick_tensor_type(np.zeros((4, 256), np.float32), el, 2) is el
        assert _pick_tensor_type(np.zeros((4, 100), np.float32), el, 2) is GGMLQuantizationType.F16


Q2_K_BLOCK_BYTES = 84  # scales(16) + qs(64) + d(2) + dmin(2)


class TestQ2K:
    """Q2_K (256-wide) write path: layout, zero handling, structure."""

    def test_block_byte_size(self, rng):
        raw = quantize_q2_k(rng.normal(size=1024).astype(np.float32))  # 4 blocks
        assert len(raw) == 4 * Q2_K_BLOCK_BYTES

    def test_zero_block_decodes_to_zero(self):
        raw = quantize_q2_k(np.zeros(Q6_K_BLOCK, dtype=np.float32))
        assert len(raw) == Q2_K_BLOCK_BYTES
        assert np.all(dequantize_q2_k(raw, Q6_K_BLOCK) == 0.0)

    def test_constant_block_is_consistent(self):
        c = 2.5
        x = np.full(Q6_K_BLOCK, c, dtype=np.float32)
        out = dequantize_q2_k(quantize_q2_k(x), x.size)
        # 2-bit grid: every element shares the same (dl, ml, q) -> one value.
        # (Q2_K's offset only covers negative mins, so an all-positive constant
        # reconstructs to a single consistent value, which may be near 0.)
        assert np.all(out == out[0])

    def test_roundtrip_is_finite(self, rng):
        x = rng.standard_normal(Q6_K_BLOCK * 2).astype(np.float32)
        out = dequantize_q2_k(quantize_q2_k(x), x.size)
        assert np.all(np.isfinite(out))

    def test_requires_multiple_of_256(self):
        with pytest.raises(ValueError, match="multiple of 256"):
            quantize_q2_k(np.zeros(100, dtype=np.float32))

    def test_writer_reader_roundtrip(self, tmp_path, rng):
        """Writer emits Q2_K; GGUFReader dequantizes it without crashing."""
        from llm.export.gguf import GGUFReader, GGUFWriter

        x = rng.standard_normal(Q6_K_BLOCK * 2).astype(np.float32)
        path = tmp_path / "q2k.gguf"
        writer = GGUFWriter(path)
        writer.add_tensor("w", x, ggml_type="q2_k")
        writer.write()
        out = GGUFReader(path).read_tensor("w")
        assert out.shape == x.shape
        assert np.all(np.isfinite(out))

    def test_exporter_resolves_and_picks_q2k(self):
        from llm.export.gguf.exporter import _pick_tensor_type, _resolve_quant_type
        from llm.export.gguf.spec import GGMLQuantizationType

        assert _resolve_quant_type("q2_k") is GGMLQuantizationType.Q2_K
        el = GGMLQuantizationType.Q2_K
        assert _pick_tensor_type(np.zeros((4, 256), np.float32), el, 2) is el
        assert _pick_tensor_type(np.zeros((4, 100), np.float32), el, 2) is GGMLQuantizationType.F16
