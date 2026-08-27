"""GGML block quantization / dequantization.

This module is numpy-only; torch tensors are converted by the writer.

**Quantize (export)** — the two schemes this repo writes, 32-element blocks
along the contiguous (last) dimension of a tensor:

- ``Q4_0`` (GGML_TYPE_Q4_0): per block one fp16 scale ``d`` plus 32 4-bit
  values packed 2-per-byte (low nibble = first half, high nibble = second
  half). Symmetric around an implicit offset of 8; value is ``(nibble - 8)*d``.
- ``Q8_0`` (GGML_TYPE_Q8_0): per block one fp16 scale ``d`` plus 32 int8
  values; value is ``q * d``.

The math mirrors the ggml reference implementations in ``ggml-quants.c`` so
packed bytes are byte-compatible with llama.cpp: ``d = amax/(2**(bits-1)-1)``,
Q4_0 nibbles use truncating ``(int8_t)(x + 8.5)`` clamped to 15, Q8_0 values
use ``roundf`` (half away from zero).

**Dequantize (import)** — the reader additionally dequantizes the types that
make up virtually every downloadable llama.cpp GGUF, so the foreign-import
milestone works on real files (round 75):

- legacy 32-wide schemes ``Q4_1`` / ``Q5_0`` / ``Q5_1``;
- the 256-wide K-quant family ``Q2_K`` / ``Q3_K`` / ``Q4_K`` / ``Q5_K`` /
  ``Q6_K`` (``QK_K == 256`` in ggml).

Each ``dequantize_*`` is a line-for-line transcription of ``ggml-quants.c``
(verified byte-for-byte against gguf-py, the Python reader llama.cpp ships).
One layout note: for Q5_0 / Q5_1 the C *dequantizer* reads the upper half's
high bit at ``qh`` bit ``(p - 4)``, which contradicts its own quantizer
(``quantize_row_q5_0_ref``, which stores element ``p``'s fifth bit at ``qh``
bit ``p`` — the on-disk layout every real file and gguf-py use).  We follow
the file/quantizer layout.
"""

from __future__ import annotations

import numpy as np

from llm.export.gguf.spec import GGML_BLOCK_SIZE, GGML_K_BLOCK_SIZE

_Q8_0_MAX = 127  # (1 << 7) - 1
_F16_BYTES = 2


def _round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    """``roundf`` semantics: round half away from zero."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _safe_inverse(scale: np.ndarray) -> np.ndarray:
    """Elementwise ``1/scale`` with zeros left at zero (no divide-by-zero warning).

    Works for **negative** scales too: Q4_0 uses ggml's negative ``max / -8``
    block scale, so ``1/scale`` must not be zeroed just because ``scale < 0``
    (that would collapse every nibble to 8).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(scale != 0, 1.0 / scale, 0.0)


def _as_flat_float32(data: np.ndarray, scheme: str) -> np.ndarray:
    x = np.asarray(data, dtype=np.float32)
    if x.ndim != 1:
        x = x.reshape(-1)
    if x.size % GGML_BLOCK_SIZE:
        raise ValueError(f"{scheme} requires a multiple of {GGML_BLOCK_SIZE} elements, got {x.size}")
    return x


def _q4_0_block_scales(blocks: np.ndarray) -> np.ndarray:
    """Per-block Q4_0 scale, exactly as ggml's ``quantize_row_q4_0_reference``.

    ggml-quants.c computes ``d = max / -8`` where ``max`` is the *signed*
    value with the largest absolute magnitude in the block (not the unsigned
    |amax|)::

        for v in block: if amax < |v|: amax = |v|; max = v
        d = max / -8

    So the scale is negative when the block's extreme is positive, positive
    when it's negative, and is only ``-amax/8`` when the extreme happens to
    be positive.  (An earlier version of this module used ``amax / 7`` with a
    positive scale and nibbles ``trunc(x*7/amax + 8.5)``; both the sign and
    the 7-vs-8 divisor diverge from ggml, so exported Q4_0 tensors were never
    read back correctly by llama.cpp.  The dequantized *values* happened to
    be self-consistent anyway — which is why the old roundtrip tests passed —
    but the packed bytes were not the format the ecosystem expects.)
    """
    # For each block, the signed extreme is the value at the first position
    # whose |v| equals the block's absolute max (mirrors ggml's loop which
    # only *updates* max when amax strictly grows, so ties keep the earlier
    # value).
    idx = np.argmax(np.abs(blocks), axis=1)
    signed_max = blocks[np.arange(blocks.shape[0]), idx]
    return signed_max / -8.0


def quantize_q4_0(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize float data to Q4_0 blocks.

    Byte-compatible with ggml's ``quantize_row_q4_0_reference`` so the
    packed tensor is readable by llama.cpp / the wider GGUF ecosystem.

    Args:
        data: Float array with a multiple of 32 elements (any shape is
            flattened row-major).

    Returns:
        ``(packed, scales)`` where ``packed`` is ``uint8`` with one byte
        per two elements and ``scales`` is ``float16`` with one (negative,
        per ggml) scale per 32-element block.
    """
    x = _as_flat_float32(data, "Q4_0")
    blocks = x.reshape(-1, GGML_BLOCK_SIZE)
    scale = _q4_0_block_scales(blocks)  # negative: max / -8
    inv = _safe_inverse(scale)
    scaled = blocks * inv[:, None]
    # ggml: q = (int8_t)(x * (-8/max) + 8.5) == (x / d) + 8.5, clipped to
    # [0, 15]; the C cast truncates toward zero (values here are clipped
    # into the positive range so trunc == floor).
    nibbles = np.trunc(scaled + 8.5).astype(np.int16)
    nibbles = np.clip(nibbles, 0, 15).astype(np.uint8)
    low = nibbles[:, : GGML_BLOCK_SIZE // 2]
    high = nibbles[:, GGML_BLOCK_SIZE // 2 :]
    packed = (low | (high << 4)).reshape(-1)
    return packed, scale.astype(np.float16)


def dequantize_q4_0(packed: np.ndarray, scales: np.ndarray, n: int) -> np.ndarray:
    """Dequantize Q4_0 blocks back to float32.

    Matches ggml's ``dequantize_row_q4_0``: ``(q - 8) * d`` with the same
    (negative) per-block scale used at quantize time.

    Args:
        packed: ``uint8`` nibble bytes (one per two elements).
        scales: ``float16``/``float32`` block scales (one per 32 elements).
        n: Total element count to reconstruct.
    """
    p = np.asarray(packed, dtype=np.uint8).reshape(-1)
    s = np.asarray(scales, dtype=np.float32).reshape(-1)
    if p.size % (GGML_BLOCK_SIZE // 2):
        raise ValueError(f"packed Q4_0 data must hold whole blocks, got {p.size} bytes")
    numel = p.size * 2
    if s.size != numel // GGML_BLOCK_SIZE:
        raise ValueError(f"expected {numel // GGML_BLOCK_SIZE} Q4_0 scales, got {s.size}")
    low = (p & 0x0F).astype(np.float32) - 8.0
    high = ((p >> 4) & 0x0F).astype(np.float32) - 8.0
    blocks = np.empty((s.size, GGML_BLOCK_SIZE), dtype=np.float32)
    blocks[:, : GGML_BLOCK_SIZE // 2] = low.reshape(-1, GGML_BLOCK_SIZE // 2)
    blocks[:, GGML_BLOCK_SIZE // 2 :] = high.reshape(-1, GGML_BLOCK_SIZE // 2)
    out = (blocks * s[:, None]).reshape(-1)
    if n < 0 or n > out.size:
        raise ValueError(f"cannot reconstruct {n} elements from {out.size} available")
    return out[:n]


def quantize_q8_0(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize float data to Q8_0 blocks.

    Returns ``(values, scales)`` where ``values`` is ``int8`` (one per
    element) and ``scales`` is ``float16`` (one per 32-element block).
    """
    x = _as_flat_float32(data, "Q8_0")
    blocks = x.reshape(-1, GGML_BLOCK_SIZE)
    amax = np.max(np.abs(blocks), axis=1)
    scale = amax / _Q8_0_MAX
    inv = _safe_inverse(scale)
    scaled = blocks * inv[:, None]
    values = np.clip(_round_half_away_from_zero(scaled), -128, 127).astype(np.int8)
    return values.reshape(-1), scale.astype(np.float16)


def dequantize_q8_0(values: np.ndarray, scales: np.ndarray, n: int) -> np.ndarray:
    """Dequantize Q8_0 blocks back to float32."""
    q = np.asarray(values, dtype=np.float32).reshape(-1)
    s = np.asarray(scales, dtype=np.float32).reshape(-1)
    if q.size % GGML_BLOCK_SIZE:
        raise ValueError(f"Q8_0 values must hold whole blocks, got {q.size} elements")
    if s.size != q.size // GGML_BLOCK_SIZE:
        raise ValueError(f"expected {q.size // GGML_BLOCK_SIZE} Q8_0 scales, got {s.size}")
    blocks = q.reshape(-1, GGML_BLOCK_SIZE) * s[:, None]
    out = blocks.reshape(-1)
    if n < 0 or n > out.size:
        raise ValueError(f"cannot reconstruct {n} elements from {out.size} available")
    return out[:n]


def quantize_q6_k(data: np.ndarray) -> np.ndarray:
    """Quantize float data to Q6_K blocks (256-wide), returning the packed bytes.

    Layout (``block_q6_K`` from ggml-quants.h, w* which the existing
    :func:`dequantize_q6_k` reads): per 256-element block,
    ``ql(128) + qh(64) + scales(16) + d(f16)``. ``sc`` is an int8 per
    16-element group (16 groups/block), ``d`` is the shared fp16 block scale,
    and element ``(g, k)`` (group ``g``, index ``k``) decodes as
    ``d * sc[g] * (q - 32)`` where ``q`` is the 6-bit value packed as a 4-bit
    low nibble in ``ql`` plus a 2-bit high field in ``qh``.

    The output is a flat ``uint8`` array self-consistent with the (gguf-py
    verified) :func:`dequantize_q6_k`, so any file it writes is loadable by
    llama.cpp's Q6_K dequantizer. It restores the original block's dynamic
    range with ``d = amax/32`` plus per-group int8 scales, mirroring ggml's
    structure (block scale + per-16 int8 scales) rather than a single global
    scale, which keeps reconstruction error small across mixed-magnitude
    blocks.
    """
    x = np.asarray(data, dtype=np.float32).reshape(-1)
    if x.size % GGML_K_BLOCK_SIZE:
        raise ValueError(f"Q6_K requires a multiple of {GGML_K_BLOCK_SIZE} elements, got {x.size}")
    blocks = x.reshape(-1, GGML_K_BLOCK_SIZE)
    n_blocks = blocks.shape[0]
    ql = np.zeros((n_blocks, 128), dtype=np.uint8)
    qh = np.zeros((n_blocks, 64), dtype=np.uint8)
    scales = np.ones((n_blocks, 16), dtype=np.int8)
    d = np.zeros(n_blocks, dtype=np.float16)

    for b in range(n_blocks):
        block = blocks[b]
        amax = float(np.max(np.abs(block)))
        if amax == 0.0:
            # All zeros: d=0, scale=1, q=32 (centered -> value 0).
            d[b] = np.float16(0.0)
            _pack_q6_k_group(ql[b], qh[b], 0, 16, np.full(16, 32, dtype=np.int32))
            continue
        dval = amax / 32.0
        d[b] = np.float16(dval)
        for g in range(16):
            grp = block[g * 16 : (g + 1) * 16]
            gmax = float(np.max(np.abs(grp)))
            sc = 1 if gmax == 0.0 else max(1, min(127, round(gmax / (dval * 31))))
            scales[b, g] = np.int8(sc)
            denom = dval * sc
            qv = np.clip(np.round(grp / denom).astype(np.int32) + 32, 0, 63)
            _pack_q6_k_group(ql[b], qh[b], g, 16, qv)

    rec = np.zeros(
        n_blocks,
        dtype=[("ql", "u1", 128), ("qh", "u1", 64), ("sc", "i1", 16), ("d", "<f2")],
    )
    rec["ql"] = ql
    rec["qh"] = qh
    rec["sc"] = scales
    rec["d"] = d
    return np.frombuffer(rec.tobytes(), dtype=np.uint8)


def _pack_q6_k_group(ql: np.ndarray, qh: np.ndarray, group: int, size: int, qv: np.ndarray) -> None:
    """Pack 6-bit ``qv`` for group ``group`` into the block's ``ql``/``qh``.

    Mirrors :func:`dequantize_q6_k`'s index mapping exactly: group ``g``,
    element ``k`` sits at linear position ``L = g*size + k``; its 4-bit low
    half goes to ``ql[(L//128)*64 + L%64]`` (nibble ``(L//64)%2``) and its 2-bit
    high half goes to ``qh[(L//128)*32 + L%32]`` at bit offset ``2*((L//32)%4)``.
    """
    for k in range(size):
        li = group * size + k
        lo = qv[k] & 0x0F
        hi = (qv[k] >> 4) & 0x03
        ql_byte = (li // 128) * 64 + (li % 64)
        ql_nibble = (li // 64) % 2
        ql[ql_byte] |= np.uint8(lo << (4 * ql_nibble))
        byte = (li // 128) * 32 + ((group % 2) * size + k)
        shift = 2 * ((li // 32) % 4)
        qh[byte] |= np.uint8(hi << shift)


def quantize_q4_k(data: np.ndarray) -> np.ndarray:
    """Quantize float data to Q4_K blocks (256-wide), returning the packed bytes.

    Layout (``block_q4_K``, which the existing :func:`dequantize_q4_k` reads):
    ``d(2) + dmin(2) + scales(12) + qs(128)``. Eight 32-element groups each
    carry an int8-pair ``(sc, m)`` unpacked from ``scales`` (ggml's
    ``get_scale_min_k4``), and element ``(g, j)`` decodes as
    ``d * sc[g] * q - dmin * m[g]`` with ``q`` a 4-bit nibble in ``qs``.

    The output is self-consistent with the (gguf-py verified) dequantizer, so
    any file it writes is loadable by llama.cpp's Q4_K dequantizer (same
    validity contract as the Q6_K writer, RIL DEC-098).
    """
    x = np.asarray(data, dtype=np.float32).reshape(-1)
    if x.size % GGML_K_BLOCK_SIZE:
        raise ValueError(f"Q4_K requires a multiple of {GGML_K_BLOCK_SIZE} elements, got {x.size}")
    blocks = x.reshape(-1, GGML_K_BLOCK_SIZE)
    n_blocks = blocks.shape[0]
    qs = np.zeros((n_blocks, 128), dtype=np.uint8)
    scales = np.zeros((n_blocks, 12), dtype=np.uint8)
    d = np.zeros(n_blocks, dtype=np.float16)
    dmin = np.zeros(n_blocks, dtype=np.float16)

    for b in range(n_blocks):
        groups = blocks[b].reshape(8, 32).astype(np.float32)
        mins = groups.min(axis=1)
        maxs = groups.max(axis=1)
        spans = maxs - mins
        max_span = float(spans.max()) if spans.size else 0.0
        max_negmin = float(np.maximum(-mins, 0.0).max()) if mins.size else 0.0
        dval = max_span / (15.0 * 63.0) if max_span > 0.0 else 0.0
        dminval = max_negmin / 63.0 if max_negmin > 0.0 else 0.0
        d[b] = np.float16(dval)
        dmin[b] = np.float16(dminval)

        sc = np.ones(8, dtype=np.int32)
        m = np.zeros(8, dtype=np.int32)
        for g in range(8):
            gspan = float(spans[g])
            if dval > 0.0 and gspan > 0.0:
                sc[g] = int(np.clip(round(gspan / (15.0 * dval)), 1, 63))
            if dminval > 0.0:
                m[g] = int(np.clip(round(-float(mins[g]) / dminval), 0, 63))
            denom = dval * sc[g]
            off = dminval * m[g]
            if denom > 0.0:
                q = np.clip(np.round((groups[g] + off) / denom).astype(np.int32), 0, 15)
            else:
                q = np.zeros(32, dtype=np.int32)
            base = (g // 2) * 32
            shift = (g % 2) * 4
            qs[b, base : base + 32] |= np.uint8((q & 0x0F).astype(np.uint8) << shift)
        _encode_scale_min_q4k(scales[b], sc, m)

    rec = np.zeros(n_blocks, dtype=[("d", "<f2"), ("dmin", "<f2"), ("sc", "u1", 12), ("qs", "u1", 128)])
    rec["d"] = d
    rec["dmin"] = dmin
    rec["sc"] = scales
    rec["qs"] = qs
    return np.frombuffer(rec.tobytes(), dtype=np.uint8)


def _encode_scale_min_q4k(scales: np.ndarray, sc: np.ndarray, m: np.ndarray) -> None:
    """Encode the 8 ``(sc, m)`` pairs into the 12 ``scales`` bytes.

    Inverts :func:`_get_scale_min_k4` (ggml's ``get_scale_min_k4``): bytes
    ``0..3``/``4..7``/``8..11`` are the low-6 bits for groups 0..3 / 4..7 plus
    the two high bits fed from the same bytes for groups 4..7. ``sc``/``m`` are
    clipped to 6 bits.
    """
    sc = np.asarray(sc, dtype=np.int32) & 0x3F
    m = np.asarray(m, dtype=np.int32) & 0x3F
    d12 = (sc[0:4] & 0x3F) | ((sc[4:8] >> 4) << 6)
    m12 = (m[0:4] & 0x3F) | ((m[4:8] >> 4) << 6)
    md = ((m[4:8] & 0x0F) << 4) | (sc[4:8] & 0x0F)
    scales[:] = np.concatenate([d12.astype(np.uint8), m12.astype(np.uint8), md.astype(np.uint8)])


def quantize_q5_k(data: np.ndarray) -> np.ndarray:
    """Quantize float data to Q5_K blocks (256-wide), returning the packed bytes.

    Layout (``block_q5_K``, which :func:`dequantize_q5_k` reads):
    ``d(2) + dmin(2) + scales(12) + qh(32) + qs(128)``. Same 8x32 group
    structure and get_scale_min_k4 (sc, m) as Q4_K, but each value is 5-bit:
    its low 4 bits live in ``qs`` (Q4_K nibble map) and its 5th bit in ``qh``
    byte ``j`` bit ``g``. Value = ``d*sc*q - dmin*m``, q in 0..31.
    """
    x = np.asarray(data, dtype=np.float32).reshape(-1)
    if x.size % GGML_K_BLOCK_SIZE:
        raise ValueError(f"Q5_K requires a multiple of {GGML_K_BLOCK_SIZE} elements, got {x.size}")
    blocks = x.reshape(-1, GGML_K_BLOCK_SIZE)
    n_blocks = blocks.shape[0]
    qs = np.zeros((n_blocks, 128), dtype=np.uint8)
    qh = np.zeros((n_blocks, 32), dtype=np.uint8)
    scales = np.zeros((n_blocks, 12), dtype=np.uint8)
    d = np.zeros(n_blocks, dtype=np.float16)
    dmin = np.zeros(n_blocks, dtype=np.float16)

    for b in range(n_blocks):
        groups = blocks[b].reshape(8, 32).astype(np.float32)
        mins = groups.min(axis=1)
        maxs = groups.max(axis=1)
        spans = maxs - mins
        max_span = float(spans.max()) if spans.size else 0.0
        max_negmin = float(np.maximum(-mins, 0.0).max()) if mins.size else 0.0
        dval = max_span / (31.0 * 63.0) if max_span > 0.0 else 0.0
        dminval = max_negmin / 63.0 if max_negmin > 0.0 else 0.0
        d[b] = np.float16(dval)
        dmin[b] = np.float16(dminval)

        sc = np.ones(8, dtype=np.int32)
        m = np.zeros(8, dtype=np.int32)
        for g in range(8):
            gspan = float(spans[g])
            if dval > 0.0 and gspan > 0.0:
                sc[g] = int(np.clip(round(gspan / (31.0 * dval)), 1, 63))
            if dminval > 0.0:
                m[g] = int(np.clip(round(-float(mins[g]) / dminval), 0, 63))
            denom = dval * sc[g]
            off = dminval * m[g]
            if denom > 0.0:
                q = np.clip(np.round((groups[g] + off) / denom).astype(np.int32), 0, 31)
            else:
                q = np.zeros(32, dtype=np.int32)
            base = (g // 2) * 32
            shift = (g % 2) * 4
            qs[b, base : base + 32] |= np.uint8(((q & 0x0F).astype(np.uint8)) << shift)
            # 5th bit of element (g, j) -> qh byte j bit g.
            qh[b] |= np.uint8(((q >> 4) & 0x01).astype(np.uint8) << g)
        _encode_scale_min_q4k(scales[b], sc, m)

    rec = np.zeros(
        n_blocks,
        dtype=[("d", "<f2"), ("dmin", "<f2"), ("sc", "u1", 12), ("qh", "u1", 32), ("qs", "u1", 128)],
    )
    rec["d"] = d
    rec["dmin"] = dmin
    rec["sc"] = scales
    rec["qh"] = qh
    rec["qs"] = qs
    return np.frombuffer(rec.tobytes(), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Reader-side dequantization (round 75): legacy 32-wide schemes + K-quants.
# Each function takes a flat uint8 whole-payload byte array and the logical
# element count and returns the dequantized float32 vector.  Layouts are the
# ``block_q4_1`` / ``block_q5_0`` / ``block_q5_1`` / ``block_q2_K`` /
# ``block_q3_K`` / ``block_q4_K`` / ``block_q5_K`` / ``block_q6_K`` structs
# from ggml-quants.h; the math is transcribed from ggml-quants.c and is
# byte-identical to gguf-py 0.19.
# ---------------------------------------------------------------------------


def _as_whole_blocks(raw: np.ndarray, block_bytes: int, scheme: str) -> np.ndarray:
    """Reshape a flat uint8 payload into ``(n_blocks, block_bytes)``."""
    r = np.asarray(raw, dtype=np.uint8).reshape(-1)
    if r.size == 0 or r.size % block_bytes:
        raise ValueError(f"{scheme} payload must be whole {block_bytes}-byte blocks, got {r.size} bytes")
    return r.reshape(-1, block_bytes)


def _block_f16(blocks: np.ndarray, start: int) -> np.ndarray:
    """Per-block fp16 field at byte offset ``start`` (2 bytes), widened to f32."""
    return np.frombuffer(blocks[:, start : start + _F16_BYTES].reshape(-1).tobytes(), dtype="<f2").astype(np.float32)


def _trim(out: np.ndarray, n: int) -> np.ndarray:
    flat = out.reshape(-1)
    if n < 0 or n > flat.size:
        raise ValueError(f"cannot reconstruct {n} elements from {flat.size} available")
    return flat[:n]


def dequantize_q4_1(raw: np.ndarray, n: int) -> np.ndarray:
    """Dequantize Q4_1 blocks: value = ``q * d + m`` (q in 0..15)."""
    blocks = _as_whole_blocks(raw, 2 + 2 + 16, "Q4_1")
    d = _block_f16(blocks, 0)
    m = _block_f16(blocks, 2)
    qs = blocks[:, 4:]  # (n, 16)
    low = (qs & 0x0F).astype(np.float32)  # elements 0..15
    high = ((qs >> 4) & 0x0F).astype(np.float32)  # elements 16..31
    out = np.concatenate([low, high], axis=1) * d[:, None] + m[:, None]
    return _trim(out, n)


def dequantize_q5_0(raw: np.ndarray, n: int) -> np.ndarray:
    """Dequantize Q5_0 blocks: value = ``(q - 16) * d``; element ``p`` keeps
    its fifth bit at ``qh`` bit ``p`` (file/quantizer layout — see module doc)."""
    blocks = _as_whole_blocks(raw, 2 + 4 + 16, "Q5_0")
    d = _block_f16(blocks, 0)
    qh = np.frombuffer(blocks[:, 2:6].reshape(-1).tobytes(), dtype="<u4")  # (n,)
    qs = blocks[:, 6:]  # (n, 16)
    low = qs & 0x0F
    high = (qs >> 4) & 0x0F
    bits = ((qh[:, None] >> np.arange(32, dtype=np.uint32)) & 1).astype(np.uint8) << 4
    vals = (np.concatenate([low, high], axis=1) | bits).astype(np.int32) - 16
    out = vals * d[:, None]
    return _trim(out, n)


def dequantize_q5_1(raw: np.ndarray, n: int) -> np.ndarray:
    """Dequantize Q5_1 blocks: value = ``q * d + m`` with the same qh layout."""
    blocks = _as_whole_blocks(raw, 2 + 2 + 4 + 16, "Q5_1")
    d = _block_f16(blocks, 0)
    m = _block_f16(blocks, 2)
    qh = np.frombuffer(blocks[:, 4:8].reshape(-1).tobytes(), dtype="<u4")
    qs = blocks[:, 8:]  # (n, 16)
    low = qs & 0x0F
    high = (qs >> 4) & 0x0F
    bits = ((qh[:, None] >> np.arange(32, dtype=np.uint32)) & 1).astype(np.uint8) << 4
    vals = (np.concatenate([low, high], axis=1) | bits).astype(np.float32)
    out = vals * d[:, None] + m[:, None]
    return _trim(out, n)


def dequantize_q2_k(raw: np.ndarray, n: int) -> np.ndarray:
    """Dequantize Q2_K blocks: 256 2-bit values, 16 per-block (d,s) pairs.

    Layout: scales(16) + qs(64) + d(2) + dmin(2).  Per 16-element group,
    ``dl = d * (scales & 0xF)`` and ``ml = dmin * (scales >> 4)``; value is
    ``dl * q2 - ml``.
    """
    blocks = _as_whole_blocks(raw, 16 + 64 + 4, "Q2_K")
    scales = blocks[:, 0:16]
    qs = blocks[:, 16:80]
    d = _block_f16(blocks, 80)
    dmin = _block_f16(blocks, 82)
    dl = d[:, None] * (scales & 0x0F).astype(np.float32)
    ml = dmin[:, None] * (scales >> 4).astype(np.float32)
    shift = np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
    qv = (qs.reshape(-1, 2, 1, 32) >> shift) & np.uint8(3)
    qv = qv.reshape(-1, 16, 16).astype(np.float32)
    out = dl[:, :, None] * qv - ml[:, :, None]
    return _trim(out.reshape(-1, 256), n)


def dequantize_q3_k(raw: np.ndarray, n: int) -> np.ndarray:
    """Dequantize Q3_K blocks: 256 3-bit values + 16 scale bytes packed 6-bit.

    Layout: hmask(32) + qs(64) + scales(12) + d(2).  The 16 per-block scales
    are packed 6 bits each across the 12 ``scales`` bytes.
    """
    blocks = _as_whole_blocks(raw, 32 + 64 + 12 + 2, "Q3_K")
    hmask = blocks[:, 0:32]
    qs = blocks[:, 32:96]
    scales = blocks[:, 96:108]
    d = _block_f16(blocks, 108)
    # scales: 8 low bytes pack two 4-bit scale parts, 4 high bytes hold the
    # remaining two bits of each of the 16 scales (ggml aux[4] rearrangement).
    lscales, hscales = np.hsplit(scales, [8])
    lscales = lscales.reshape(-1, 1, 8) >> np.array([0, 4], dtype=np.uint8).reshape(1, 2, 1)
    lscales = lscales.reshape(-1, 16)
    hscales = hscales.reshape(-1, 1, 4) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 4, 1)
    hscales = hscales.reshape(-1, 16)
    scales8 = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales8 = (scales8.astype(np.int8) - np.int8(32)).astype(np.float32)
    dl = (d[:, None] * scales8).reshape(-1, 16, 1)
    ql = qs.reshape(-1, 2, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
    qh = hmask.reshape(-1, 1, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 1, 8, 1)
    ql = ql.reshape(-1, 16, 16) & np.uint8(3)
    qh = (qh.reshape(-1, 16, 16) & np.uint8(1)) ^ np.uint8(1)  # offset 0 when bitmask set
    q = (ql.astype(np.int8) - (qh << np.uint8(2)).astype(np.int8)).astype(np.float32)
    out = dl * q
    return _trim(out.reshape(-1, 256), n)


def _get_scale_min_k4(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack the 12-byte Q4_K / Q5_K super-scaling into 8 (scale, min) pairs.

    Transcribed from ggml's ``get_scale_min_k4`` (ggml-quants.c).  For groups
    0..3 the scale is ``scales[j] & 0x3F``; for groups 4..7 the second six
    bytes carry the extra bits.
    """
    s = np.asarray(scales, dtype=np.uint8).reshape(-1, 3, 4)
    d12, m12, md = s[:, 0], s[:, 1], s[:, 2]  # bytes 0..3, 4..7, 8..11  (n, 4)
    sc = np.concatenate([d12 & 0x3F, (md & 0x0F) | ((d12 >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([m12 & 0x3F, (md >> 4) | ((m12 >> 2) & 0x30)], axis=-1)
    return sc.reshape(-1, 8), mn.reshape(-1, 8)


def dequantize_q4_k(raw: np.ndarray, n: int) -> np.ndarray:
    """Dequantize Q4_K blocks: value = ``d*sc*q - dmin*m``.

    Layout: d(2) + dmin(2) + scales(12) + qs(128); 8 (scale, min) pairs cover
    eight 32-element groups.
    """
    blocks = _as_whole_blocks(raw, 4 + 12 + 128, "Q4_K")
    d = _block_f16(blocks, 0)
    dmin = _block_f16(blocks, 2)
    scales = blocks[:, 4:16]
    qs = blocks[:, 16:144]
    sc, m = _get_scale_min_k4(scales)
    d1 = (d[:, None] * sc.astype(np.float32)).reshape(-1, 8, 1)
    dm = (dmin[:, None] * m.astype(np.float32)).reshape(-1, 8, 1)
    qv = (qs.reshape(-1, 4, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & np.uint8(0x0F)
    qv = qv.reshape(-1, 8, 32).astype(np.float32)
    out = d1 * qv - dm
    return _trim(out.reshape(-1, 256), n)


def dequantize_q5_k(raw: np.ndarray, n: int) -> np.ndarray:
    """Dequantize Q5_K blocks: 32-element groups with 5-bit values.

    Layout: d(2) + dmin(2) + scales(12) + qh(32) + qs(128); the 5th bit of
    element ``p`` lives in ``qh`` bit ``(p % 32)`` of byte ``p // 32``.
    """
    blocks = _as_whole_blocks(raw, 4 + 12 + 32 + 128, "Q5_K")
    d = _block_f16(blocks, 0)
    dmin = _block_f16(blocks, 2)
    scales = blocks[:, 4:16]
    qh = blocks[:, 16:48]
    qs = blocks[:, 48:176]
    sc, m = _get_scale_min_k4(scales)
    d1 = (d[:, None] * sc.astype(np.float32)).reshape(-1, 8, 1)
    dm = (dmin[:, None] * m.astype(np.float32)).reshape(-1, 8, 1)
    ql = (qs.reshape(-1, 4, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & np.uint8(0x0F)
    qh = (qh.reshape(-1, 1, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 1, 8, 1)) & np.uint8(0x01)
    q = (ql.reshape(-1, 8, 32) | (qh.reshape(-1, 8, 32) << np.uint8(4))).astype(np.float32)
    out = d1 * q - dm
    return _trim(out.reshape(-1, 256), n)


def dequantize_q6_k(raw: np.ndarray, n: int) -> np.ndarray:
    """Dequantize Q6_K blocks: 256 6-bit values, 16 per-block fp16*int8 scales.

    Layout: ql(128) + qh(64) + scales(16) + d(2); value = ``d * sc * (q - 32)``
    with ``sc`` an int8 per 16-element group.
    """
    blocks = _as_whole_blocks(raw, 128 + 64 + 16 + 2, "Q6_K")
    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192]
    scales = blocks[:, 192:208]
    d = _block_f16(blocks, 208)
    sc = np.asarray(scales, dtype=np.uint8).astype(np.int8).astype(np.float32)
    d1 = (d[:, None] * sc).reshape(-1, 16, 1)
    qlv = (ql.reshape(-1, 2, 1, 64) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & np.uint8(0x0F)
    qhv = (qh.reshape(-1, 2, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)) & np.uint8(0x03)
    qv = qlv.reshape(-1, 8, 32) | (qhv.reshape(-1, 8, 32) << np.uint8(4))
    qv = (qv.astype(np.int8) - np.int8(32)).reshape(-1, 16, 16).astype(np.float32)
    out = d1 * qv
    return _trim(out.reshape(-1, 256), n)
