"""Typed GGUF metadata key-value encoding (spec §Metadata).

GGUF metadata is a sequence of ``(key: GGUFString, type: uint32, value)``
triples. This module owns the value-type inference (writer side) and the
binary encode/decode for every ``GGUFValueType``, including ``ARRAY``.
It is dependency-free so the format layer stays importable without
numpy/torch.
"""

from __future__ import annotations

import struct
from typing import Any

from llm.export.gguf.spec import GGUFError, GGUFValueType

_FIXED = {
    GGUFValueType.UINT8: struct.Struct("<B"),
    GGUFValueType.INT8: struct.Struct("<b"),
    GGUFValueType.UINT16: struct.Struct("<H"),
    GGUFValueType.INT16: struct.Struct("<h"),
    GGUFValueType.UINT32: struct.Struct("<I"),
    GGUFValueType.INT32: struct.Struct("<i"),
    GGUFValueType.FLOAT32: struct.Struct("<f"),
    GGUFValueType.UINT64: struct.Struct("<Q"),
    GGUFValueType.INT64: struct.Struct("<q"),
    GGUFValueType.FLOAT64: struct.Struct("<d"),
}


def _coerce_scalar(value: Any) -> Any:
    """Normalize numpy scalars (``np.int64``, ``np.float32``, ``np.bool_``…)
    to their Python-native equivalent so GGUF type inference sees a real
    ``int``/``float``/``bool``.

    Numpy scalars are not ``isinstance`` instances of the Python builtins
    (``np.int64(5)`` is not an ``int``), so the eager type checks below
    rejected them with ``GGUFError`` — an export abort whenever metadata
    derived from a config or checkpoint field happened to be a numpy scalar
    (GGUF deep-dive finding #6). This file stays dependency-free: it probes
    the scalar protocol (``ndim == 0`` + ``item()``) instead of importing
    numpy. 0-d ``torch.Tensor`` scalars share the same protocol and are
    coerced the same way.
    """
    ndim = getattr(value, "ndim", None)
    item = getattr(value, "item", None)
    if ndim == 0 and callable(item):
        try:
            return item()
        except TypeError, ValueError, OverflowError:
            # Not coercible as a plain scalar (e.g. a 0-d object array) —
            # leave it for the value-type checks to reject with context.
            return value
    return value


def value_type(value: Any) -> GGUFValueType:
    """Infer the GGUF value type for a Python value (writer-side inference).

    Mapping: ``bool`` → BOOL, ``str`` → STRING, ``int`` → INT64,
    ``float`` → FLOAT32, homogeneous ``list``/``tuple`` → ARRAY.
    """
    value = _coerce_scalar(value)
    if isinstance(value, bool):
        return GGUFValueType.BOOL
    if isinstance(value, str):
        return GGUFValueType.STRING
    if isinstance(value, int):
        return GGUFValueType.INT64
    if isinstance(value, float):
        return GGUFValueType.FLOAT32
    if isinstance(value, (list, tuple)):
        if not value:
            return GGUFValueType.ARRAY
        elem_type = value_type(value[0])
        if elem_type == GGUFValueType.ARRAY:
            raise GGUFError("nested arrays are not supported in GGUF metadata")
        if any(value_type(item) != elem_type for item in value[1:]):
            raise GGUFError(f"metadata arrays must be homogeneous, got {value!r}")
        return GGUFValueType.ARRAY
    raise GGUFError(f"cannot encode metadata value {value!r} (type {type(value).__name__})")


def encode_string(value: str) -> bytes:
    """Encode a ``GGUFString``: uint64 byte length + UTF-8 payload."""
    payload = value.encode("utf-8")
    return struct.pack("<Q", len(payload)) + payload


def decode_string(payload: bytes) -> tuple[str, int]:
    """Decode a ``GGUFString`` from the front of ``payload``.

    Returns ``(value, consumed_bytes)``. Raises :class:`GGUFError` on
    truncation or an implausible length.
    """
    if len(payload) < 8:
        raise GGUFError("truncated GGUF string header")
    (length,) = struct.unpack_from("<Q", payload, 0)
    end = 8 + length
    if end > len(payload):
        raise GGUFError(f"truncated GGUF string ({length} bytes declared, {len(payload) - 8} available)")
    return payload[8:end].decode("utf-8"), end


def encode_value(value: Any) -> tuple[GGUFValueType, bytes]:
    """Encode a Python value into ``(type_code, payload_bytes)``."""
    t = value_type(value)
    if t == GGUFValueType.BOOL:
        return t, struct.pack("<B", int(value))
    if t == GGUFValueType.STRING:
        return t, encode_string(value)
    if t == GGUFValueType.INT64:
        return t, struct.pack("<q", value)
    if t == GGUFValueType.FLOAT32:
        return t, struct.pack("<f", value)
    if t == GGUFValueType.ARRAY:
        if not value:
            # GGUF needs a concrete element type even for empty arrays; UINT8 is a neutral choice.
            return t, struct.pack("<II", 0, GGUFValueType.UINT8)
        elem_type = value_type(value[0])
        body = b"".join(encode_value(item)[1] for item in value)
        return t, struct.pack("<II", len(value), elem_type) + body
    raise AssertionError(f"unhandled value type {t}")  # pragma: no cover


def decode_value(type_code: int, payload: bytes) -> tuple[Any, int]:
    """Decode one GGUF value from the front of ``payload``.

    Returns ``(value, consumed_bytes)``. Raises :class:`GGUFError` for
    unknown type codes and truncated payloads.
    """
    try:
        t = GGUFValueType(type_code)
    except ValueError:
        raise GGUFError(f"unknown GGUF value type code {type_code}") from None

    if t == GGUFValueType.BOOL:
        if len(payload) < 1:
            raise GGUFError("truncated BOOL metadata value")
        return bool(payload[0]), 1
    if t == GGUFValueType.STRING:
        return decode_string(payload)
    if t in _FIXED:
        fmt = _FIXED[t]
        if len(payload) < fmt.size:
            raise GGUFError(f"truncated {t.name} metadata value ({len(payload)} bytes)")
        return fmt.unpack_from(payload, 0)[0], fmt.size
    if t == GGUFValueType.ARRAY:
        if len(payload) < 8:
            raise GGUFError("truncated ARRAY metadata header")
        count, elem_type = struct.unpack_from("<II", payload, 0)
        pos = 8
        out: list[Any] = []
        for _ in range(count):
            elem, consumed = decode_value(elem_type, payload[pos:])
            out.append(elem)
            pos += consumed
        return out, pos
    raise AssertionError(f"unhandled value type {t}")  # pragma: no cover


def encode_metadata(metadata: dict[str, Any]) -> bytes:
    """Serialize an ordered metadata dict into GGUF KV bytes."""
    chunks: list[bytes] = []
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"metadata key must be a non-empty string, got {key!r}")
        t, payload = encode_value(value)
        chunks.append(encode_string(key) + struct.pack("<I", t) + payload)
    return b"".join(chunks)


def decode_metadata(data: bytes, count: int) -> dict[str, Any]:
    """Deserialize ``count`` GGUF KV pairs from ``data`` into an ordered dict."""
    out: dict[str, Any] = {}
    pos = 0
    for _ in range(count):
        key, consumed = decode_string(data[pos:])
        pos += consumed
        if len(data) - pos < 4:
            raise GGUFError("truncated metadata value type")
        (type_code,) = struct.unpack_from("<I", data, pos)
        pos += 4
        value, consumed = decode_value(type_code, data[pos:])
        out[key] = value
        pos += consumed
    return out
