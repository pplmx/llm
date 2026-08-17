"""GGUF reader (GGUF v3 container).

The reader parses the whole file eagerly (v1; no mmap) and exposes the
header, the typed metadata dict, and per-tensor info plus two accessors:
raw payload bytes (:meth:`GGUFReader.read_tensor_raw`) and the
dequantized float32 view (:meth:`GGUFReader.read_tensor`).

Malformed files — wrong magic, unsupported version, truncation,
out-of-bounds offsets, unknown type codes — raise :class:`GGUFError`
with a message naming the offending record.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

from llm.export.gguf.metadata import decode_value
from llm.export.gguf.quant import (
    dequantize_q2_k,
    dequantize_q3_k,
    dequantize_q4_0,
    dequantize_q4_1,
    dequantize_q4_k,
    dequantize_q5_0,
    dequantize_q5_1,
    dequantize_q5_k,
    dequantize_q6_k,
    dequantize_q8_0,
)
from llm.export.gguf.spec import (
    GGML_BLOCK_SIZE,
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_HEADER_SIZE,
    GGUF_MAGIC,
    GGUF_VERSION,
    SUPPORTED_TENSOR_TYPES,
    GGMLQuantizationType,
    GGUFError,
    GGUFHeader,
    GGUFTensorInfo,
    align_up,
    tensor_data_size,
)

# Reader-side dequantizers for the block-quantized types beyond Q4_0/Q8_0:
# legacy 32-wide schemes (Q4_1/Q5_0/Q5_1) and the 256-wide K-quant family
# (Q2_K..Q6_K) that dominate real llama.cpp files (round 75 milestone).  Each
# takes the flat uint8 payload and returns the dequantized float32 vector.
_READER_DEQUANTIZERS = {
    GGMLQuantizationType.Q4_1: dequantize_q4_1,
    GGMLQuantizationType.Q5_0: dequantize_q5_0,
    GGMLQuantizationType.Q5_1: dequantize_q5_1,
    GGMLQuantizationType.Q2_K: dequantize_q2_k,
    GGMLQuantizationType.Q3_K: dequantize_q3_k,
    GGMLQuantizationType.Q4_K: dequantize_q4_k,
    GGMLQuantizationType.Q5_K: dequantize_q5_k,
    GGMLQuantizationType.Q6_K: dequantize_q6_k,
}

_MAX_STRING = 1 << 24
_MAX_RANK = 64


def _read_string_at(data: bytes, pos: int) -> tuple[str, int]:
    """Read a ``GGUFString`` at ``pos``; returns ``(value, new_pos)``."""
    if pos + 8 > len(data):
        raise GGUFError("truncated GGUF string header")
    (length,) = struct.unpack_from("<Q", data, pos)
    start = pos + 8
    end = start + length
    if length > _MAX_STRING:
        raise GGUFError(f"implausible GGUF string length {length}")
    if end > len(data):
        raise GGUFError(f"truncated GGUF string ({length} bytes declared, {len(data) - start} available)")
    return data[start:end].decode("utf-8"), end


class GGUFReader:
    """Parse and read a GGUF file.

    Attributes:
        path: The source file path.
        header: Parsed :class:`GGUFHeader`.
        metadata: Ordered metadata dict (typed Python values).
        tensors: ``name -> GGUFTensorInfo`` mapping in file order.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        data = self.path.read_bytes()
        if len(data) < GGUF_HEADER_SIZE:
            raise GGUFError(f"{self.path}: file too small to be GGUF ({len(data)} bytes)")
        magic, version, tensor_count, kv_count = struct.unpack_from("<IIQQ", data, 0)
        if magic != GGUF_MAGIC:
            raise GGUFError(f"{self.path}: bad magic 0x{magic:08x}, not a GGUF file")
        if not 1 <= version <= GGUF_VERSION:
            raise GGUFError(f"{self.path}: unsupported GGUF version {version} (supported 1..{GGUF_VERSION})")
        self.header = GGUFHeader(
            magic=magic,
            version=version,
            tensor_count=tensor_count,
            metadata_kv_count=kv_count,
        )

        pos = GGUF_HEADER_SIZE
        metadata: dict[str, object] = {}
        for _ in range(kv_count):
            key, pos = _read_string_at(data, pos)
            if pos + 4 > len(data):
                raise GGUFError(f"metadata key {key!r}: truncated value type")
            (type_code,) = struct.unpack_from("<I", data, pos)
            pos += 4
            value, consumed = decode_value(type_code, data[pos:])
            metadata[key] = value
            pos += consumed
        self.metadata = metadata

        tensors: dict[str, GGUFTensorInfo] = {}
        for _ in range(tensor_count):
            name, pos = _read_string_at(data, pos)
            if pos + 4 > len(data):
                raise GGUFError(f"tensor {name!r}: truncated dimension count")
            (n_dims,) = struct.unpack_from("<I", data, pos)
            pos += 4
            if n_dims > _MAX_RANK:
                raise GGUFError(f"tensor {name!r}: implausible rank {n_dims}")
            if pos + 8 * n_dims > len(data):
                raise GGUFError(f"tensor {name!r}: truncated dimensions")
            dims = struct.unpack_from(f"<{n_dims}Q", data, pos)
            pos += 8 * n_dims
            if pos + 12 > len(data):
                raise GGUFError(f"tensor {name!r}: truncated type/offset")
            (type_code, offset) = struct.unpack_from("<IQ", data, pos)
            pos += 12
            try:
                ttype = GGMLQuantizationType(type_code)
            except ValueError:
                raise GGUFError(f"tensor {name!r}: unknown GGML type code {type_code}") from None
            if ttype not in SUPPORTED_TENSOR_TYPES:
                raise GGUFError(
                    f"tensor {name!r}: unsupported GGML type {ttype.name} ({type_code}); "
                    f"v1 supports {sorted(t.name for t in SUPPORTED_TENSOR_TYPES)}"
                )
            shape = tuple(reversed(dims))
            tensors[name] = GGUFTensorInfo(
                name=name,
                shape=shape,
                ggml_type=ttype,
                offset=offset,
                data_size=tensor_data_size(ttype, shape),
            )
        self.tensors = tensors

        self._data_start = align_up(pos, GGUF_DEFAULT_ALIGNMENT)
        self._data = data
        for info in tensors.values():
            if info.offset < self._data_start:
                raise GGUFError(
                    f"tensor {info.name!r}: offset {info.offset} precedes the data section start {self._data_start}"
                )
            if info.offset + info.data_size > len(data):
                raise GGUFError(
                    f"tensor {info.name!r}: data range {info.offset}..{info.offset + info.data_size} "
                    f"exceeds file size {len(data)}"
                )

    def _info(self, name: str) -> GGUFTensorInfo:
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError(f"no tensor named {name!r} in {self.path}") from None

    def read_tensor_raw(self, name: str) -> bytes:
        """Return the exact on-disk payload bytes for ``name`` (no dequantization)."""
        info = self._info(name)
        return bytes(self._data[info.offset : info.offset + info.data_size])

    def read_tensor(self, name: str) -> np.ndarray:
        """Read and dequantize ``name`` into a float32 array of its logical shape.

        F32/F16 payloads are returned as-is (F16 widened); Q4_0/Q8_0 and the
        reader-side legacy/K-quant types (Q4_1/Q5_0/Q5_1, Q2_K..Q6_K) are
        dequantized with the reference ggml math.
        """
        info = self._info(name)
        raw = self._data[info.offset : info.offset + info.data_size]
        if info.ggml_type == GGMLQuantizationType.F32:
            return np.frombuffer(raw, dtype="<f4").reshape(info.shape)
        if info.ggml_type == GGMLQuantizationType.F16:
            return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(info.shape)

        numel = math.prod(info.shape)
        if numel == 0:
            # An empty quantized tensor has no blocks to dequantize; mirror
            # the F32/F16 empty-array return instead of raising from the
            # block parser (round-75 review LOW).
            return np.empty(info.shape, dtype=np.float32)
        if info.ggml_type in _READER_DEQUANTIZERS:
            return _READER_DEQUANTIZERS[info.ggml_type](np.frombuffer(raw, dtype=np.uint8), numel).reshape(info.shape)

        block_count = numel // GGML_BLOCK_SIZE
        # ggml block layout is interleaved per 32-element block: a 2-byte
        # fp16 scale followed by the packed values (16 bytes for Q4_0, 32
        # for Q8_0). llama.cpp / gguf-py emit and read this layout; the
        # reader must de-interleave it back into per-block scales + body.
        data_per_block = GGML_BLOCK_SIZE // 2 if info.ggml_type == GGMLQuantizationType.Q4_0 else GGML_BLOCK_SIZE
        block_bytes = 2 + data_per_block
        buf = np.frombuffer(raw, dtype=np.uint8).reshape(block_count, block_bytes)
        scales = np.frombuffer(buf[:, :2].reshape(-1).tobytes(), dtype="<f2").astype(np.float32)
        body = buf[:, 2:].reshape(-1).tobytes()
        if info.ggml_type == GGMLQuantizationType.Q4_0:
            return dequantize_q4_0(np.frombuffer(body, dtype=np.uint8), scales, numel).reshape(info.shape)
        if info.ggml_type == GGMLQuantizationType.Q8_0:
            return dequantize_q8_0(np.frombuffer(body, dtype=np.int8), scales, numel).reshape(info.shape)
        raise GGUFError(f"tensor {name!r}: unsupported GGML type {info.ggml_type.name}")  # pragma: no cover
