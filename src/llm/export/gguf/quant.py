"""Q4_0 / Q8_0 GGML block quantization.

Both schemes quantize 32-element blocks along the contiguous (last)
dimension of a tensor:

- ``Q4_0`` (GGML_TYPE_Q4_0): per block one fp16 scale ``d`` plus 32
  4-bit values packed 2-per-byte (low nibble = first half, high nibble =
  second half). Symmetric around an implicit offset of 8; dequantized
  value is ``(nibble - 8) * d``.
- ``Q8_0`` (GGML_TYPE_Q8_0): per block one fp16 scale ``d`` plus 32
  int8 values; dequantized value is ``q * d``.

The math mirrors the ggml reference implementations in
``ggml-quants.c`` so packed bytes are byte-compatible with llama.cpp:
``d = amax / (2**(bits-1) - 1)``, Q4_0 nibbles use truncating
``(int8_t)(x + 8.5)`` clamped to 15, Q8_0 values use ``roundf``
(half away from zero).

This module is numpy-only; torch tensors are converted by the writer.
"""

from __future__ import annotations

import numpy as np

from llm.export.gguf.spec import GGML_BLOCK_SIZE

_Q8_0_MAX = 127  # (1 << 7) - 1


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
