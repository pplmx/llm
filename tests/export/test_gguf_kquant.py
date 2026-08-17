"""GGUF K-quant + legacy-quant dequantization (import path) — round 75 milestone.

The GGUF reader (rounds 71-73) imports foreign llama.cpp files but only
understood F32 / F16 / Q4_0 / Q8_0; every real downloadable llama.cpp GGUF is
a 256-element K-quant (Q4_K_M / Q5_K_M / Q6_K / Q3_K_M / Q2_K) or a legacy
32-element type (Q4_1 / Q5_0 / Q5_1).  Round 72's import milestone was
therefore in practice limited to F16 files.  This module pins the
dequantization of eight new tensor types against the canonical llama.cpp
implementations and exercises a full foreign import on a K-quant file.

Ground truth: the dequantizers are line-for-line transcriptions of
``ggml-quants.c`` (ggml-org/ggml), cross-checked here against ``gguf-py``
(the official llama.cpp Python reader) on random and structured block
payloads — the two must produce identical float32.  Where gguf-py also
provides a quantizer (Q4_1 / Q5_0 / Q5_1), a full quantize→dequantize
round-trip bounds the import error.

Layout note (file truth vs the C *dequantizer*): ggml stores the Q5_0 / Q5_1
high bit of element ``p`` at ``qh`` bit ``p`` (what ``quantize_row_q5_0_ref``
writes and gguf-py's reader reads); the C ``dequantize_row_q5_0`` in
ggml-quants.c reads bit ``(p - 4)`` for the upper half, which disagrees with
its own quantizer.  We follow the file/quantizer layout, i.e. gguf-py.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from llm.export.gguf import GGUFReader, load_gguf_model
from llm.export.gguf.metadata import encode_metadata
from llm.export.gguf.quant import (
    dequantize_q2_k,
    dequantize_q3_k,
    dequantize_q4_1,
    dequantize_q4_k,
    dequantize_q5_0,
    dequantize_q5_1,
    dequantize_q5_k,
    dequantize_q6_k,
)
from llm.export.gguf.spec import (
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_HEADER_SIZE,
    GGUF_MAGIC,
    GGUF_VERSION,
    GGMLQuantizationType,
    align_up,
)
from llm.runtime.bootstrap import ensure_builtins_registered
from llm.runtime.model_factory import ModelFactory
from llm.training.core.config import ModelConfig

# (GGML type, block size, block byte size)
_BLOCK_LAYOUTS = {
    GGMLQuantizationType.Q4_1: (32, 20),
    GGMLQuantizationType.Q5_0: (32, 22),
    GGMLQuantizationType.Q5_1: (32, 24),
    GGMLQuantizationType.Q2_K: (256, 84),
    GGMLQuantizationType.Q3_K: (256, 110),
    GGMLQuantizationType.Q4_K: (256, 144),
    GGMLQuantizationType.Q5_K: (256, 176),
    GGMLQuantizationType.Q6_K: (256, 210),
}

DEQUANT = {
    GGMLQuantizationType.Q4_1: dequantize_q4_1,
    GGMLQuantizationType.Q5_0: dequantize_q5_0,
    GGMLQuantizationType.Q5_1: dequantize_q5_1,
    GGMLQuantizationType.Q2_K: dequantize_q2_k,
    GGMLQuantizationType.Q3_K: dequantize_q3_k,
    GGMLQuantizationType.Q4_K: dequantize_q4_k,
    GGMLQuantizationType.Q5_K: dequantize_q5_k,
    GGMLQuantizationType.Q6_K: dequantize_q6_k,
}


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


def _random_payload(rng, qtype: GGMLQuantizationType, n_blocks: int, *, structured: bool):
    """Random or structured ``uint8`` byte payload for ``n_blocks`` blocks."""
    _, block_bytes = _BLOCK_LAYOUTS[qtype]
    if structured:
        # Alternating byte values exercise both nibble nibbles / bit positions
        # without relying on chance.
        base = np.tile(np.arange(block_bytes, dtype=np.uint8) * 37, n_blocks)
        return base.reshape(n_blocks, block_bytes)
    return rng.integers(0, 256, size=(n_blocks, block_bytes), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Cross-check against gguf-py (the official llama.cpp Python reader)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "qtype",
    [
        GGMLQuantizationType.Q4_1,
        GGMLQuantizationType.Q5_0,
        GGMLQuantizationType.Q5_1,
        GGMLQuantizationType.Q2_K,
        GGMLQuantizationType.Q3_K,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K,
    ],
    ids=lambda t: t.name,
)
def test_dequant_matches_gguf_py_reference(qtype, rng):
    """Our dequantizer must be byte-for-byte the gguf-py reference.

    Both are independent transcriptions of ggml-quants.c; feeding identical
    raw block bytes to both must produce identical float32.  This is the
    strongest available ground truth for the *file* layout (gguf-py is the
    reader llama.cpp ships).
    """
    quants = pytest.importorskip("gguf.quants")
    block_size, _ = _BLOCK_LAYOUTS[qtype]
    n_blocks = 8
    numel = n_blocks * block_size
    fn = DEQUANT[qtype]

    for structured in (True, False):
        payload = _random_payload(rng, qtype, n_blocks, structured=structured)
        ours = fn(payload.reshape(-1), numel)
        theirs = quants.dequantize(payload, qtype).reshape(-1)
        assert ours.shape == theirs.shape == (numel,)
        # NaN-aware equality: random fp16 scale fields can be NaN/inf; both
        # implementations reproduce the same NaN at the same positions.
        assert np.array_equal(ours, theirs, equal_nan=True), (
            f"{qtype.name} ({'structured' if structured else 'random'}) diverged from gguf-py — "
            f"first diff index {np.flatnonzero(~(ours == theirs) & ~(np.isnan(ours) & np.isnan(theirs)))[0] if True else -1}"
        )


@pytest.mark.parametrize(
    "qtype",
    [GGMLQuantizationType.Q4_1, GGMLQuantizationType.Q5_0, GGMLQuantizationType.Q5_1],
    ids=lambda t: t.name,
)
def test_legacy_quantize_roundtrip_within_reference_error(qtype, rng):
    """gguf-py quantizes these legacy types; our dequant must invert it."""
    quants = pytest.importorskip("gguf.quants")
    block_size, _ = _BLOCK_LAYOUTS[qtype]
    fn = DEQUANT[qtype]
    x = rng.normal(size=block_size * 8).astype(np.float32)
    packed = quants.quantize(x, qtype)  # (n_blocks, block_bytes) uint8
    back = fn(packed.reshape(-1), x.size).astype(np.float32)
    assert np.max(np.abs(back - x)) < 1.0, f"{qtype.name} roundtrip error unbounded"


# ---------------------------------------------------------------------------
# Always-running structural tests (no gguf-py required)
# ---------------------------------------------------------------------------


def _pack_q6_k(q6: np.ndarray, scales16: np.ndarray, d: float) -> bytes:
    """Pack 256 6-bit codes into one block_q6_K (210 bytes) — ggml layout.

    Layout (ggml block_q6_K): ql[128] (4-bit lows), qh[64] (2-bit highs),
    scales[16] (int8, one per 16 element subgroup in the C's walk order),
    d (fp16).  Transcribed from dequantize_row_q6_K element map:

      elem n+l    <- ql[l]&0xF        | (qh[l]>>0 &3)<<4   scale is
      elem n+32+l <- ql[l+32]&0xF     | (qh[l]>>2 &3)<<4   scale is+2
      elem n+64+l <- (ql[l]>>4)       | (qh[l]>>4 &3)<<4   scale is+4
      elem n+96+l <- (ql[l+32]>>4)    | (qh[l]>>6 &3)<<4   scale is+6
    where n in {0, 128}, is = l//16, and q-lows indexed against elem n+64+l.
    """
    q = np.asarray(q6, dtype=np.uint16).reshape(256)
    ql = np.zeros(128, dtype=np.uint8)
    qh = np.zeros(64, dtype=np.uint8)
    for ci, n in enumerate((0, 128)):
        ql_base = 64 * ci  # ql advances 64 bytes per 128-chunk (dequant: ql += 64)
        qh_base = 32 * ci  # qh advances 32 bytes per chunk
        for idx in range(32):
            e0, e1, e2, e3 = n + idx, n + 32 + idx, n + 64 + idx, n + 96 + idx
            ql[ql_base + idx] = (q[e0] & 0x0F) | ((q[e2] & 0x0F) << 4)
            ql[ql_base + 32 + idx] = (q[e1] & 0x0F) | ((q[e3] & 0x0F) << 4)
            qh[qh_base + idx] = (
                ((q[e0] >> 4) & 0x03)
                | (((q[e1] >> 4) & 0x03) << 2)
                | (((q[e2] >> 4) & 0x03) << 4)
                | (((q[e3] >> 4) & 0x03) << 6)
            )
    hdr = np.frombuffer(np.float16(d).tobytes(), dtype=np.uint8)
    return bytes(np.concatenate([ql, qh, np.asarray(scales16, dtype=np.int8).view(np.uint8), hdr]))


def test_q6_k_block_roundtrip_internal():
    """The test's own packer must agree with our dequant (independence)."""
    d = 1.0
    q6 = np.array([32 + (i % 8) - 4 for i in range(256)], dtype=np.uint16)
    scales = np.full(16, 3, dtype=np.int8)
    raw = _pack_q6_k(q6, scales, d)
    out = dequantize_q6_k(np.frombuffer(raw, dtype=np.uint8), 256)
    expected = (q6.astype(np.float32) - 32.0) * d * 3.0
    assert np.array_equal(out, expected), f"{np.max(np.abs(out - expected))}"


def test_q6_k_zero_block_is_zero():
    out = dequantize_q6_k(np.zeros(210, dtype=np.uint8), 256)
    assert np.array_equal(out, np.zeros(256))


def test_q5_0_high_bit_position_matches_file_layout():
    """Element p's 5th bit lives at qh bit p (quantizer/file layout)."""
    # d = 1.0, qh = 0, qs byte0 = 0x12 -> elem0 = 2-16, elem16 = 1-16.
    raw = bytearray(22)
    raw[0:2] = np.float16(1.0).tobytes()
    raw[6] = 0x12
    out = dequantize_q5_0(np.frombuffer(bytes(raw), dtype=np.uint8), 32)
    assert out[0] == 2 - 16
    assert out[16] == 1 - 16
    # Set qh bit 16 -> element 16 gains +16 (element p <-> bit p).
    raw[2:6] = struct.pack("<I", 1 << 16)
    out = dequantize_q5_0(np.frombuffer(bytes(raw), dtype=np.uint8), 32)
    assert out[16] == (1 + 16) - 16
    assert out[0] == 2 - 16  # bit 0 untouched


def test_q5_1_affine_offset():
    raw = bytearray(24)
    raw[0:2] = np.float16(1.0).tobytes()
    raw[2:4] = np.float16(2.0).tobytes()  # m = 2
    raw[8] = 0x12  # qs byte0 (d, m, qh, qs layout)
    out = dequantize_q5_1(np.frombuffer(bytes(raw), dtype=np.uint8), 32)
    assert out[0] == 2 * 1.0 + 2.0
    assert out[16] == 1 * 1.0 + 2.0


def test_q4_1_affine_offset():
    raw = bytearray(20)
    raw[0:2] = np.float16(1.0).tobytes()  # d = 1
    raw[2:4] = np.float16(1.0).tobytes()  # m = 1
    raw[4] = 0x12
    out = dequantize_q4_1(np.frombuffer(bytes(raw), dtype=np.uint8), 32)
    assert out[0] == 2 * 1.0 + 1.0
    assert out[16] == 1 * 1.0 + 1.0


def test_q4_k_uniform_scale_block():
    # d=1, dmin=0, scales bytes d12=[1,1,1,1], m12=[0]*4, md=[1,1,1,1]
    # -> all eight sc=1, all m=0 (get_scale_min_k4); out = q nibble.
    raw = bytearray(144)
    raw[0:2] = np.float16(1.0).tobytes()
    raw[2:4] = np.float16(0.0).tobytes()  # dmin = 0
    raw[4:8] = b"\x01\x01\x01\x01"  # d12 -> sc[0..3] = 1
    raw[8:12] = b"\x00\x00\x00\x00"  # m12 -> m[0..3] = 0
    raw[12:16] = b"\x01\x01\x01\x01"  # md  -> sc[4..7] = 1
    raw[16] = 0x12  # first qs byte: elem 0 = 2, elem 32 = 1
    out = dequantize_q4_k(np.frombuffer(bytes(raw), dtype=np.uint8), 256)
    assert out[0] == 2
    assert out[32] == 1
    assert out[1] == 0  # untouched nibbles
    assert out[255] == 0


def test_each_type_rejects_partial_payload():
    for qtype, fn in DEQUANT.items():
        _, block_bytes = _BLOCK_LAYOUTS[qtype]
        with pytest.raises(ValueError, match="block"):
            fn(np.zeros(block_bytes - 1, dtype=np.uint8), 0)


def test_each_type_rejects_out_of_range_count():
    block_size, block_bytes = _BLOCK_LAYOUTS[GGMLQuantizationType.Q6_K]
    with pytest.raises(ValueError, match="elements"):
        dequantize_q6_k(np.zeros(block_bytes, dtype=np.uint8), block_size + 1)


# ---------------------------------------------------------------------------
# Reader: a hand-built GGUF with a K-quant tensor round-trips through GGUFReader
# ---------------------------------------------------------------------------


def _build_raw_gguf(
    tensors: list[tuple[str, GGMLQuantizationType, tuple[int, ...], bytes]],
    metadata: dict | None = None,
) -> bytes:
    """Encapsulate raw payload bytes into a GGUF v3 container (test helper).

    Mirrors GGUFWriter.write() but accepts pre-packed payload bytes so
    K-quant files (which the writer cannot emit) can be exercised by the
    reader.  ``tensors`` is ``(name, ttype, logical_shape, payload_bytes)``.
    """
    metadata = metadata or {}
    meta_b = encode_metadata(metadata)
    payloads = [(name, ttype, shape, data) for name, ttype, shape, data in tensors]
    infos_size = 0
    for name, _, shape, _ in payloads:
        name_b = name.encode("utf-8")
        infos_size += 8 + len(name_b) + 4 + 8 * len(shape) + 12
    data_start = align_up(GGUF_HEADER_SIZE + len(meta_b) + infos_size, GGUF_DEFAULT_ALIGNMENT)
    offset = data_start
    buf = bytearray()
    buf += struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(payloads), len(metadata))
    buf += meta_b
    for name, ttype, shape, data in payloads:
        name_b = name.encode("utf-8")
        dims = tuple(reversed(shape))
        buf += (
            struct.pack("<Q", len(name_b))
            + name_b
            + struct.pack("<I", len(dims))
            + struct.pack(f"<{len(dims)}Q", *dims)
            + struct.pack("<IQ", ttype.value, offset)
        )
        offset += align_up(len(data), GGUF_DEFAULT_ALIGNMENT)
    buf += b"\x00" * (data_start - len(buf))
    for _, _, _, data in payloads:
        buf += data
        buf += b"\x00" * (align_up(len(data), GGUF_DEFAULT_ALIGNMENT) - len(data))
    return bytes(buf)


def test_reader_dequantizes_q4_k_and_q6_k(tmp_path):
    # Build an 8-block Q4_K payload.  d=1, dmin=0, all-ones scales (bytes
    # [4:8] = 1 -> sc[0..3] = 1) -> identity nibble decode for groups 0..3.
    q4_k = np.zeros((8, 144), dtype=np.uint8)
    for b in range(8):
        q4_k[b, 0:2] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
        q4_k[b, 4:8] = 1
        q4_k[b, 16] = 0x12  # first qs byte: low nibble 2 -> elem 0, high 1 -> elem 32
    q6_k = _pack_q6_k(np.full(256, 48, dtype=np.uint16), np.full(16, 1, dtype=np.int8), 0.5)
    q6_k_tensor = b"".join([q6_k] * 4)  # 4 blocks

    data = _build_raw_gguf(
        [
            ("w_q4k", GGMLQuantizationType.Q4_K, (8, 256), q4_k.tobytes()),
            ("w_q6k", GGMLQuantizationType.Q6_K, (4, 256), q6_k_tensor),
        ]
    )
    path = tmp_path / "kquant.gguf"
    path.write_bytes(data)
    reader = GGUFReader(path)

    q4 = reader.read_tensor("w_q4k")
    assert q4.shape == (8, 256)
    assert q4[0, 0] == 2
    assert q4[0, 32] == 1

    q6 = reader.read_tensor("w_q6k")
    assert q6.shape == (4, 256)
    # q6=48, d=0.5, scale=1 -> 0.5 * (48-32) = 8
    assert np.allclose(q6, 8.0)


def test_reader_empty_kquant_tensor_returns_empty(tmp_path):
    """A (0, N) K-quant tensor reads back as an empty array, not an error
    (round-75 review LOW: previously raised a raw ValueError from the block
    parser)."""
    data = _build_raw_gguf([("w_q4k", GGMLQuantizationType.Q4_K, (0, 256), b"")])
    path = tmp_path / "empty_q4k.gguf"
    path.write_bytes(data)
    reader = GGUFReader(path)
    q = reader.read_tensor("w_q4k")
    assert q.shape == (0, 256)
    assert q.dtype == np.float32
    assert q.size == 0


# ---------------------------------------------------------------------------
# End-to-end: a foreign llama.cpp file with K-quantized weights imports
# ---------------------------------------------------------------------------


def _build_llama_cfg() -> ModelConfig:
    """Tiny Llama-style model whose hidden/intermediate dims are 256-block friendly."""
    return ModelConfig(
        vocab_size=32,
        hidden_size=256,
        num_layers=1,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=512,
        max_seq_len=16,
        use_glu=True,
        mlp_activation="silu",
        norm_impl="rms_norm",
        norm_first=True,
        qkv_bias=False,
        mlp_bias=False,
        lm_head_bias=False,
        use_rope=True,
        rope_theta=10000.0,
    )


def _quantize_q6_k_row(x: np.ndarray) -> np.ndarray:
    """Reference Q6_K quantizer (valid encoding, not byte-identical to ggml).

    q := round(x/d) + 32 (0..63), d = block_amax/31, all per-16 scales = 1.
    Dequant is ``d * 1 * (q - 32)``, so the extremes ±amax map to q 63/1.
    """
    x = np.asarray(x, dtype=np.float32)
    blocks = x.reshape(-1, 256)
    out = []
    for b in blocks:
        amax = float(np.abs(b).max())
        if amax == 0.0:
            out.append(_pack_q6_k(np.full(256, 32, dtype=np.uint16), np.ones(16, dtype=np.int8), 0.0))
            continue
        d = np.float16(amax / 31.0)
        q = np.clip(np.rint(b / float(d)).astype(np.int64) + 32, 0, 63).astype(np.uint16)
        out.append(_pack_q6_k(q, np.ones(16, dtype=np.int8), d))
    return np.frombuffer(b"".join(out), dtype=np.uint8)


def _quantize_q5_k_row(x: np.ndarray) -> np.ndarray:
    """Reference Q5_K quantizer (valid encoding, not byte-identical to ggml).

    Same min-anchored scheme as Q4_K but 5-bit codes: d = (max-min)/31,
    dmin = -min, all sc/m = 1.  qh is a 32-byte field shared by all four
    64-element groups: byte ``l`` holds the 5th bits of elements ``e`` with
    ``e % 32 == l`` at bit ``e // 32`` (bit 2g+half per group/quarter).
    """
    x = np.asarray(x, dtype=np.float32)
    blocks = x.reshape(-1, 256)
    out = []
    for b in blocks:
        minv = float(b.min())
        span = float(b.max()) - minv
        d = np.float16(span / 31.0) if span > 0 else np.float16(0.0)
        q = np.zeros(256, dtype=np.uint8)
        if span > 0:
            q = np.clip(np.rint((b - minv) / float(d)).astype(np.int64), 0, 31).astype(np.uint8)
        dmin = np.float16(-minv)
        ql = np.zeros(128, dtype=np.uint8)
        qh = np.zeros(32, dtype=np.uint8)
        for g in range(4):
            for idx in range(32):
                e0 = g * 64 + idx
                e1 = g * 64 + 32 + idx
                ql[g * 32 + idx] = (q[e0] & 0x0F) | ((q[e1] & 0x0F) << 4)
                if q[e0] & 0x10:
                    qh[idx] |= 1 << (2 * g)
                if q[e1] & 0x10:
                    qh[idx] |= 1 << (2 * g + 1)
        block = bytearray(176)
        block[0:2] = d.tobytes()  # d
        block[2:4] = dmin.tobytes()  # dmin = -min
        block[4:8] = b"\x01\x01\x01\x01"  # d12 -> sc[0..3] = 1
        block[8:12] = b"\x01\x01\x01\x01"  # m12 -> m[0..3] = 1
        block[12:16] = b"\x11\x11\x11\x11"  # md -> m[4..7] = 1 and sc[4..7] = 1
        block[16:48] = qh.tobytes()
        block[48:176] = ql.tobytes()
        out.append(bytes(block))
    return np.frombuffer(b"".join(out), dtype=np.uint8)


def _quantize_q4_k_row(x: np.ndarray) -> np.ndarray:
    """Reference Q4_K quantizer (valid encoding, not byte-identical to ggml).

    Anchors the block range with the per-block min: d = (max-min)/15,
    dmin = -min, all per-group scales/min = 1, so dequant ``d*q - dmin``
    reproduces the values within one quantum.  Negative blocks need dmin != 0.
    """
    x = np.asarray(x, dtype=np.float32)
    blocks = x.reshape(-1, 256)
    out = []
    for b in blocks:
        minv = float(b.min())
        span = float(b.max()) - minv
        d = np.float16(span / 15.0) if span > 0 else np.float16(0.0)
        q = np.zeros(256, dtype=np.uint8)
        if span > 0:
            q = np.clip(np.rint((b - minv) / float(d)).astype(np.int64), 0, 15).astype(np.uint8)
        dmin = np.float16(-minv)
        qs = np.zeros(128, dtype=np.uint8)
        for g in range(4):
            for idx in range(32):
                qs[g * 32 + idx] = q[g * 64 + idx] | (q[g * 64 + 32 + idx] << 4)
        block = bytearray(144)
        block[0:2] = d.tobytes()  # d
        block[2:4] = dmin.tobytes()  # dmin = -min
        block[4:8] = b"\x01\x01\x01\x01"  # d12 -> sc[0..3] = 1
        block[8:12] = b"\x01\x01\x01\x01"  # m12 -> m[0..3] = 1
        block[12:16] = b"\x11\x11\x11\x11"  # md -> m[4..7] = 1 and sc[4..7] = 1
        block[16:144] = qs.tobytes()
        out.append(bytes(block))
    return np.frombuffer(b"".join(out), dtype=np.uint8)


@pytest.mark.parametrize("kquant", ["q6_k", "q5_k", "q4_k"])
def test_foreign_import_kquant_within_quantizer_error(tmp_path, kquant):
    """A llama.cpp-style file whose matmuls are K-quant imports correctly.

    Norms and embeddings stay F16 (exactly how llama.cpp stores them); only
    the matmul weights are block-quantized — the realistic Q4_K_M/Q5_K_M/Q6_K
    layout that round-72's import milestone could not previously read.
    """
    cfg = _build_llama_cfg()
    ensure_builtins_registered()
    torch.manual_seed(11 if kquant == "q6_k" else 12)
    model = ModelFactory.from_config(cfg, norm_eps=1e-5).eval()

    # Reuse the canonical llama.cpp tensor renaming (splits fused qkv).
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    tensors: list[tuple[str, GGMLQuantizationType, tuple[int, ...], bytes]] = []
    header = {
        "general.architecture": "llama",
        "general.name": f"tiny-{kquant}",
        "general.quantization_version": 2,
        "llama.context_length": cfg.max_seq_len,
        "llama.embedding_length": cfg.hidden_size,
        "llama.block_count": cfg.num_layers,
        "llama.feed_forward_length": cfg.intermediate_size,
        "llama.attention.head_count": cfg.num_heads,
        "llama.attention.head_count_kv": cfg.num_kv_heads,
        "llama.attention.layer_norm_rms_epsilon": 1e-5,
        "llama.rope.freq_base": cfg.rope_theta,
        "llama.vocab_size": cfg.vocab_size,
    }

    head_dim = model.transformer_blocks[0].self_attn.head_dim
    kv_dim = model.transformer_blocks[0].self_attn.kv_dim
    q_size = model.num_heads * head_dim

    def add(name, data: np.ndarray, ttype, logical_shape: tuple[int, ...] | None = None):
        """Register a tensor; fp16 arrays are serialized, K-quant bytes passed as-is."""
        if ttype is GGMLQuantizationType.F16:
            payload = np.ascontiguousarray(data, dtype="<f2").tobytes()
            shape = tuple(data.shape)
        else:  # caller already passed pre-packed uint8 K-quant payload bytes
            payload = np.ascontiguousarray(data, dtype=np.uint8).tobytes()
            shape = tuple(logical_shape or data.shape)
        tensors.append((name, ttype, shape, payload))

    # Norm/embedding heads stay F16 bytes (real llama.cpp layout).
    add("token_embd.weight", sd["embedding_layer.token_embeddings.weight"], GGMLQuantizationType.F16)
    add("output_norm.weight", sd["final_norm.weight"], GGMLQuantizationType.F16)
    add("output.weight", sd["lm_head.weight"], GGMLQuantizationType.F16)
    p = "transformer_blocks.0"
    add("blk.0.attn_norm.weight", sd[f"{p}.norm1.weight"], GGMLQuantizationType.F16)
    add("blk.0.ffn_norm.weight", sd[f"{p}.norm2.weight"], GGMLQuantizationType.F16)

    # Matmul weights -> K-quant where the last dim is block-size friendly.
    quantizers = {
        "q6_k": (GGMLQuantizationType.Q6_K, _quantize_q6_k_row),
        "q5_k": (GGMLQuantizationType.Q5_K, _quantize_q5_k_row),
        "q4_k": (GGMLQuantizationType.Q4_K, _quantize_q4_k_row),
    }
    qtype, kq = quantizers[kquant]
    for llama_name, state_key in {
        "blk.0.attn_output.weight": f"{p}.self_attn.out_proj.weight",
        "blk.0.ffn_gate.weight": f"{p}.mlp.fc1.weight",
        "blk.0.ffn_up.weight": f"{p}.mlp.gate_proj.weight",
        "blk.0.ffn_down.weight": f"{p}.mlp.fc2.weight",
    }.items():
        add(llama_name, kq(sd[state_key]), qtype, sd[state_key].shape)
    qkv = sd[f"{p}.self_attn.qkv_proj.weight"]
    for llama_name, slice_ in (
        ("blk.0.attn_q.weight", qkv[:q_size]),
        ("blk.0.attn_k.weight", qkv[q_size : q_size + kv_dim]),
        ("blk.0.attn_v.weight", qkv[q_size + kv_dim :]),
    ):
        add(llama_name, kq(slice_), qtype, slice_.shape)

    path = tmp_path / f"foreign_{kquant}.gguf"
    path.write_bytes(_build_raw_gguf(tensors, header))
    restored = load_gguf_model(path)

    original = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
    recovered = {k: v.detach().float().cpu() for k, v in restored.state_dict().items()}
    assert set(original) == set(recovered)

    for name in original:
        if name in ("token_embeddings.weight",):  # sanity — never quantized
            continue
        max_abs = float(original[name].abs().max())
        # Block quantum: Q6_K covers 2*max_abs in 64 levels (<= max_abs/31),
        # Q5_K / Q4_K cover [min,max] in 32/16 levels (span <= 2*max_abs so
        # <= max_abs/15).  Plus the fp16 scale/offset rounding and the F16
        # norms' own half-precision error.
        quantum = max_abs / (31 if kquant == "q6_k" else 15)
        bound = max(1e-3, quantum + max_abs / 128 + 1e-4)
        err = float((original[name] - recovered[name]).abs().max())
        assert err <= bound, f"{name}: err {err} > bound {bound} (max_abs {max_abs})"

    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out = restored(ids)
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert torch.isfinite(logits).all()
