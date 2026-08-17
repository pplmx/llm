"""GGUF writer (GGUF v3 container).

The writer assembles header + metadata + tensor info + 32-byte-aligned
tensor data in memory and flushes it atomically (temp file + rename).
Tensor data is stored in row-major order; block-quantized types (Q4_0 /
Q8_0) quantize 32-element blocks along the last dimension.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Any

import numpy as np

from llm.export.gguf.metadata import encode_metadata
from llm.export.gguf.quant import quantize_q4_0, quantize_q8_0
from llm.export.gguf.spec import (
    EXPORT_TENSOR_TYPES,
    GGML_BLOCK_SIZE,
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_HEADER_SIZE,
    GGUF_MAGIC,
    GGUF_VERSION,
    GGMLQuantizationType,
    GGUFError,
    align_up,
    can_quantize_shape,
    parse_ggml_type,
)


def _as_float32_array(data: Any) -> np.ndarray:
    """Coerce ``data`` (numpy array or torch tensor) to a contiguous float32 array.

    Torch tensors are duck-typed via ``detach``/``float``/``cpu`` so this
    module stays numpy-only; CUDA tensors and ``requires_grad`` tensors
    are handled transparently.
    """
    if hasattr(data, "detach"):
        data = data.detach().float().cpu()
    arr = np.asarray(data)
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"GGUF writer v1 only supports floating-point tensors, got dtype {arr.dtype}")
    arr = arr.astype(np.float32, copy=False)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def _interleave_blocks(scales: np.ndarray, data: np.ndarray) -> bytes:
    """Serialize ggml block tensors with the on-disk **interleaved** layout.

    ggml stores each 32-element block as a fixed-size C struct,
    ``[fp16 d][packed values]`` — ``block_q4_0`` is 18 bytes (2 fp16 scale +
    16 nibble bytes), ``block_q8_0`` is 34 bytes (2 fp16 scale + 32 int8).
    llama.cpp reads a quantized GGUF tensor by casting the whole payload to
    ``block_q4_0*``/``block_q8_0*`` and walking blocks, so scale and data
    must interleave **per block**. Storing all scales then all data — a
    layout that round-trips within this repo — misaligns every block for the
    wider GGUF ecosystem: llama.cpp reads block *k*'s scale from byte
    ``k*18`` (inside the data region) and produces garbage weights.
    """
    data_per_block = data.size // scales.size
    dtype = np.dtype([("d", "<f2"), ("qs", ("u1", data_per_block))])
    rec = np.zeros(scales.shape[0], dtype=dtype)
    rec["d"] = scales.astype(np.float16)
    rec["qs"] = np.asarray(data, dtype=np.uint8).reshape(scales.shape[0], data_per_block)
    return rec.tobytes()


def _encode_payload(arr: np.ndarray, ttype: GGMLQuantizationType) -> bytes:
    """Serialize a flat float32 array into the GGUF payload for ``ttype``."""
    flat = arr.reshape(-1)
    if ttype == GGMLQuantizationType.F32:
        return flat.astype("<f4", copy=False).tobytes()
    if ttype == GGMLQuantizationType.F16:
        return flat.astype("<f2", copy=False).tobytes()
    if ttype == GGMLQuantizationType.Q4_0:
        packed, scales = quantize_q4_0(flat)
        return _interleave_blocks(scales, packed)
    if ttype == GGMLQuantizationType.Q8_0:
        values, scales = quantize_q8_0(flat)
        return _interleave_blocks(scales, values)
    raise GGUFError(f"unsupported GGML tensor type {ttype.name}")  # pragma: no cover


def _tensor_info_size(name: str, shape: tuple[int, ...]) -> int:
    """Serialized size of one tensor info record (name + dims + type + offset)."""
    return 8 + len(name.encode("utf-8")) + 4 + 8 * len(shape) + 4 + 8


def _encode_tensor_info(name: str, shape: tuple[int, ...], ttype: GGMLQuantizationType, offset: int) -> bytes:
    """Serialize one tensor info record; dimensions are stored reversed."""
    name_b = name.encode("utf-8")
    dims = tuple(reversed(shape))
    return (
        struct.pack("<Q", len(name_b))
        + name_b
        + struct.pack("<I", len(dims))
        + struct.pack(f"<{len(dims)}Q", *dims)
        + struct.pack("<IQ", ttype.value, offset)
    )


class GGUFWriter:
    """Incremental GGUF v3 writer.

    Usage::

        writer = GGUFWriter("model.gguf")
        writer.add_metadata("general.name", "tiny")
        writer.add_tensor("w", weight_numpy, ggml_type="q8_0")
        path = writer.write()
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        version: int = GGUF_VERSION,
        alignment: int = GGUF_DEFAULT_ALIGNMENT,
    ) -> None:
        if not 1 <= version <= GGUF_VERSION:
            raise ValueError(f"unsupported GGUF version {version} (supported 1..{GGUF_VERSION})")
        if alignment <= 0:
            raise ValueError(f"alignment must be positive, got {alignment}")
        if alignment < GGUF_DEFAULT_ALIGNMENT:
            # The GGUF spec fixes tensor-data alignment at
            # ``GGUF_DEFAULT_ALIGNMENT`` (32); the reader hardcodes 32 for
            # ``_data_start``. A smaller writer alignment would emit a file
            # whose own reader (and llama.cpp) rejects every tensor as
            # preceding the data section (RIL ISS-059). Reject it up front.
            raise ValueError(f"alignment must be >= GGUF_DEFAULT_ALIGNMENT ({GGUF_DEFAULT_ALIGNMENT}), got {alignment}")
        self.output_path = Path(output_path)
        self.version = version
        self.alignment = alignment
        self._metadata: dict[str, Any] = {}
        self._tensors: list[tuple[str, GGMLQuantizationType, tuple[int, ...], bytes]] = []

    def add_metadata(self, key: str, value: Any) -> None:
        """Register one metadata KV pair (later duplicates overwrite)."""
        if not isinstance(key, str) or not key:
            raise ValueError(f"metadata key must be a non-empty string, got {key!r}")
        encode_metadata({key: value})  # validate encodability early
        self._metadata[key] = value

    def add_tensor(
        self,
        name: str,
        data: Any,
        ggml_type: GGMLQuantizationType | str,
    ) -> None:
        """Register one tensor with an explicit GGML type.

        Args:
            name: Tensor name (must be unique).
            data: ``numpy`` array or ``torch`` tensor of floating dtype.
            ggml_type: One of ``F32`` / ``F16`` / ``Q4_0`` / ``Q8_0``
                (or a case-insensitive name like ``"q8_0"``).

        Raises:
            ValueError: For duplicate names, non-float input, or a
                block-quantized type whose last dimension is not a
                multiple of 32.
            GGUFError: For unsupported tensor types.
        """
        if not isinstance(name, str) or not name:
            raise ValueError(f"tensor name must be a non-empty string, got {name!r}")
        if any(existing == name for existing, _, _, _ in self._tensors):
            raise ValueError(f"duplicate tensor name {name!r}")
        ttype = ggml_type if isinstance(ggml_type, GGMLQuantizationType) else parse_ggml_type(str(ggml_type))
        if ttype not in EXPORT_TENSOR_TYPES:
            raise GGUFError(
                f"{ttype.name} is reader-supported but not exportable; "
                f"the writer emits {sorted(t.name for t in EXPORT_TENSOR_TYPES)}"
            )
        arr = _as_float32_array(data)
        if arr.ndim == 0:
            raise ValueError(f"tensor {name!r}: scalar tensors are not supported")
        if arr.size == 0:
            # An empty tensor is never a legitimate weight. In the quantized
            # types it also crashed with a raw ``ZeroDivisionError``
            # (``data_per_block = data.size // scales.size`` → ``0 // 0``)
            # deep inside block serialization (GGUF deep-dive finding #3).
            # Reject it up front so the export fails with a clear message.
            raise ValueError(f"tensor {name!r}: empty tensors (0 elements) are not supported")
        shape = tuple(int(d) for d in arr.shape)
        if ttype in (GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q8_0) and not can_quantize_shape(shape):
            raise ValueError(
                f"tensor {name!r}: {ttype.name} requires the last dimension to be a multiple of "
                f"{GGML_BLOCK_SIZE} (got shape {shape})"
            )
        payload = _encode_payload(arr, ttype)
        self._tensors.append((name, ttype, shape, payload))

    def write(self) -> Path:
        """Assemble and atomically write the GGUF file; returns the output path."""
        metadata_bytes = encode_metadata(self._metadata)
        infos_size = sum(_tensor_info_size(name, shape) for name, _, shape, _ in self._tensors)
        data_start = align_up(GGUF_HEADER_SIZE + len(metadata_bytes) + infos_size, self.alignment)

        offset = data_start
        entries: list[tuple[str, GGMLQuantizationType, tuple[int, ...], bytes, int]] = []
        for name, ttype, shape, payload in self._tensors:
            entries.append((name, ttype, shape, payload, offset))
            offset += align_up(len(payload), self.alignment)

        buf = io.BytesIO()
        buf.write(
            struct.pack(
                "<IIQQ",
                GGUF_MAGIC,
                self.version,
                len(self._tensors),
                len(self._metadata),
            )
        )
        buf.write(metadata_bytes)
        for name, ttype, shape, _, off in entries:
            buf.write(_encode_tensor_info(name, shape, ttype, off))
        buf.write(b"\x00" * (data_start - buf.tell()))
        for _, _, _, payload, _ in entries:
            buf.write(payload)
            buf.write(b"\x00" * (align_up(len(payload), self.alignment) - len(payload)))

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.output_path.with_name(self.output_path.name + ".tmp")
        tmp_path.write_bytes(buf.getvalue())
        tmp_path.replace(self.output_path)
        return self.output_path
