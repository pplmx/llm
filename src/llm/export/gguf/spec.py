"""GGUF format constants and structural dataclasses.

Implements the GGUF v3 container layout (header + metadata + tensor info)
as specified by the GGML project (``docs/gguf.md`` in ggml-org/ggml):

- header: magic ``0x46554747`` ("GGUF"), ``uint32`` version, ``uint64``
  tensor count, ``uint64`` metadata KV count;
- metadata: typed key-value pairs (see :mod:`llm.export.gguf.metadata`);
- tensor info: name, dimension count, dimensions (stored in reverse
  order), GGML quantization type, and the absolute byte offset of the
  tensor payload;
- tensor data: 32-byte-aligned payloads, each padded back to the
  alignment after its bytes.

This module is the single source of truth for the format's numbers and
layouts and is dependency-free (no numpy/torch), so the format layer can
be imported and reasoned about in isolation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

# "GGUF" as a little-endian uint32 (bytes 47 47 55 46).
GGUF_MAGIC = 0x4655_4747
# Current GGUF container version (mirrors ``GGUF_VERSION`` in gguf-py).
GGUF_VERSION = 3
# Default tensor-data alignment in bytes (``GGUF_DEFAULT_ALIGNMENT``).
GGUF_DEFAULT_ALIGNMENT = 32
# Serialized header size: magic u32 + version u32 + tensor count u64 + KV count u64.
GGUF_HEADER_SIZE = 24
# Element count per block for the block-quantized types (QK4_0 / QK8_0).
GGML_BLOCK_SIZE = 32


class GGUFError(ValueError):
    """Raised when a GGUF file, metadata blob, or tensor payload is malformed or unsupported."""


class GGUFValueType(IntEnum):
    """GGUF metadata value types (spec §Value Types)."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLQuantizationType(IntEnum):
    """GGML tensor data types as stored in GGUF tensor info (``ggml_type``).

    The integer type codes were renumbered by ggml PR #6050; the four
    types implemented in v1 (``F32`` / ``F16`` / ``Q4_0`` / ``Q8_0``)
    have stable codes across all versions. The remaining values follow
    the current ggml.h enumeration.
    """

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28


@dataclass(frozen=True)
class GGUFHeader:
    """Parsed GGUF header."""

    magic: int
    version: int
    tensor_count: int
    metadata_kv_count: int


@dataclass(frozen=True)
class GGUFTensorInfo:
    """Parsed GGUF tensor info.

    ``shape`` is the LOGICAL shape in row-major (PyTorch/NumPy) order —
    e.g. ``(out_features, in_features)``. GGUF stores dimensions in the
    reverse order on disk; reader and writer translate at the boundary.
    """

    name: str
    shape: tuple[int, ...]
    ggml_type: GGMLQuantizationType
    offset: int
    data_size: int


# Element byte size for unquantized types.
_RAW_TYPE_SIZES = {
    GGMLQuantizationType.F32: 4,
    GGMLQuantizationType.F16: 2,
}

# Per-block byte size for the block-quantized types implemented in v1:
# fp16 scale + packed payload (16 nibble bytes for Q4_0, 32 int8 bytes for Q8_0).
_BLOCK_TYPE_SIZES = {
    GGMLQuantizationType.Q4_0: 2 + 16,
    GGMLQuantizationType.Q8_0: 2 + 32,
}

SUPPORTED_TENSOR_TYPES = frozenset(
    {GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q8_0}
)


def tensor_data_size(t: GGMLQuantizationType | int, shape: Sequence[int]) -> int:
    """Return the on-disk payload size in bytes for ``t`` and a logical ``shape``."""
    ttype = GGMLQuantizationType(t)
    numel = math.prod(shape)
    if ttype in _RAW_TYPE_SIZES:
        return numel * _RAW_TYPE_SIZES[ttype]
    if ttype in _BLOCK_TYPE_SIZES:
        if numel % GGML_BLOCK_SIZE:
            raise GGUFError(
                f"{ttype.name} requires a multiple of {GGML_BLOCK_SIZE} elements, got {numel} (shape {tuple(shape)})"
            )
        return (numel // GGML_BLOCK_SIZE) * _BLOCK_TYPE_SIZES[ttype]
    raise GGUFError(
        f"unsupported GGML tensor type {ttype.name} ({ttype.value}); "
        f"v1 supports {sorted(t.name for t in SUPPORTED_TENSOR_TYPES)}"
    )


def is_quantized(t: GGMLQuantizationType | int) -> bool:
    """Return True if the type is one of the block-quantized v1 schemes."""
    return GGMLQuantizationType(t) in _BLOCK_TYPE_SIZES


def can_quantize_shape(shape: Sequence[int]) -> bool:
    """Return True if a tensor with this shape can be block-quantized (last dim % 32 == 0)."""
    return bool(shape) and shape[-1] % GGML_BLOCK_SIZE == 0


def parse_ggml_type(name: str) -> GGMLQuantizationType:
    """Parse a case-insensitive type name such as ``"f16"`` or ``"Q4_0"``."""
    try:
        return GGMLQuantizationType[name.upper()]
    except KeyError:
        raise ValueError(f"unknown GGML type {name!r}") from None


def align_up(n: int, alignment: int) -> int:
    """Round ``n`` up to the next multiple of ``alignment``."""
    return -(-n // alignment) * alignment
