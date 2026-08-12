"""GGUF writer/reader round-trips, byte layout, and corrupt-file handling."""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from llm.export.gguf import GGUFReader, GGUFWriter
from llm.export.gguf.metadata import encode_value
from llm.export.gguf.quant import quantize_q4_0, quantize_q8_0
from llm.export.gguf.spec import (
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_HEADER_SIZE,
    GGUF_MAGIC,
    GGUF_VERSION,
    GGMLQuantizationType,
    GGUFError,
    align_up,
)


def _tensor_info(
    name: bytes = b"w",
    n_dims: int = 1,
    dims: tuple[int, ...] = (64,),
    type_code: int = 8,
    offset: int = 0,
) -> bytes:
    """Encode a tensor-info record; ``n_dims`` may exceed ``len(dims)`` to simulate truncation."""
    return (
        struct.pack("<Q", len(name))
        + name
        + struct.pack("<I", n_dims)
        + struct.pack(f"<{len(dims)}Q", *dims)
        + struct.pack("<I", type_code)
        + struct.pack("<Q", offset)
    )


def _assemble(head: bytes, payload: bytes = b"") -> bytes:
    """Pad head to 32 bytes, then append payload + padding."""
    data_start = align_up(len(head), GGUF_DEFAULT_ALIGNMENT)
    padded = align_up(len(payload), GGUF_DEFAULT_ALIGNMENT)
    return head + b"\x00" * (data_start - len(head)) + payload + b"\x00" * (padded - len(payload))


def _header(magic: int = GGUF_MAGIC, version: int = GGUF_VERSION, tensors: int = 1, kvs: int = 0) -> bytes:
    return struct.pack("<IIQQ", magic, version, tensors, kvs)


class TestWriterHeader:
    """The file starts with the spec header."""

    def test_header_fields(self, tmp_path):
        writer = GGUFWriter(tmp_path / "h.gguf")
        writer.add_metadata("k", 1)
        writer.add_tensor("w", np.zeros(64, dtype=np.float32), ggml_type="q8_0")
        path = writer.write()

        data = path.read_bytes()
        magic, version, tensor_count, kv_count = struct.unpack_from("<IIQQ", data, 0)
        assert magic == GGUF_MAGIC
        assert version == GGUF_VERSION
        assert tensor_count == 1
        assert kv_count == 1

    def test_writer_returns_path_and_creates_parents(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "model.gguf"
        result = GGUFWriter(path).write()
        assert result == path
        assert path.exists()

    def test_invalid_version_rejected(self):
        with pytest.raises(ValueError, match="unsupported GGUF version"):
            GGUFWriter("x.gguf", version=99)

    def test_sub32_alignment_rejected(self):
        """Regression (RIL ISS-059): alignment < 32 (the GGUF tensor-data
        floor) must be rejected at construction.

        The writer computes ``data_start`` with its custom ``alignment`` but
        the reader hardcodes 32 for ``_data_start`` and rejects every tensor
        whose offset precedes it — a writer alignment below 32 emits a file
        that its own reader (and llama.cpp) cannot read."""
        with pytest.raises(ValueError, match="alignment"):
            GGUFWriter("x.gguf", alignment=16)


class TestMetadataRoundTrip:
    def test_all_supported_values(self, tmp_path):
        metadata = {
            "general.name": "tiny",
            "general.file_type": 1,
            "general.float": 0.5,
            "general.flag": True,
            "general.tags": ["a", "b"],
            "general.empty": [],
            "general.ints": [1, 2, 3],
        }
        writer = GGUFWriter(tmp_path / "m.gguf")
        for key, value in metadata.items():
            writer.add_metadata(key, value)
        writer.write()

        reader = GGUFReader(tmp_path / "m.gguf")
        assert reader.metadata == metadata

    def test_metadata_order_preserved(self, tmp_path):
        writer = GGUFWriter(tmp_path / "m.gguf")
        writer.add_metadata("z", 1)
        writer.add_metadata("a", 2)
        writer.add_metadata("m", 3)
        writer.write()
        assert list(GGUFReader(tmp_path / "m.gguf").metadata) == ["z", "a", "m"]

    def test_empty_metadata(self, tmp_path):
        GGUFWriter(tmp_path / "m.gguf").write()
        assert GGUFReader(tmp_path / "m.gguf").metadata == {}

    def test_bad_key_raises(self):
        writer = GGUFWriter("x.gguf")
        with pytest.raises(ValueError, match="metadata key"):
            writer.add_metadata("", 1)


class TestTensorRoundTrip:
    @pytest.mark.parametrize(
        ("type_name", "type_enum"),
        [("f32", GGMLQuantizationType.F32), ("f16", GGMLQuantizationType.F16)],
    )
    def test_raw_types_exact(self, tmp_path, type_name: str, type_enum):
        # Integer values are exactly representable in both F32 and F16,
        # so the round-trip must be bit-exact for either raw type.
        data = np.arange(256, dtype=np.float32).reshape(4, 64)
        writer = GGUFWriter(tmp_path / "raw.gguf")
        writer.add_tensor("w", data, ggml_type=type_name)
        writer.write()

        reader = GGUFReader(tmp_path / "raw.gguf")
        assert reader.tensors["w"].ggml_type is type_enum
        assert reader.tensors["w"].shape == (4, 64)
        np.testing.assert_array_equal(reader.read_tensor("w"), data)

    @pytest.mark.parametrize("type_name", ["q4_0", "q8_0"])
    def test_quantized_roundtrip_within_tolerance(self, tmp_path, type_name: str):
        rng = np.random.default_rng(11)
        data = rng.normal(size=(4, 64)).astype(np.float32)
        writer = GGUFWriter(tmp_path / "q.gguf")
        writer.add_tensor("w", data, ggml_type=type_name)
        writer.write()

        reader = GGUFReader(tmp_path / "q.gguf")
        recovered = reader.read_tensor("w")
        quantize = quantize_q4_0 if type_name == "q4_0" else quantize_q8_0
        _, scales = quantize(data.reshape(-1))
        assert np.max(np.abs(recovered - data)) <= float(np.max(scales)) + 1e-6

    def test_quantized_raw_bytes_match_quantizer(self, tmp_path):
        rng = np.random.default_rng(3)
        data = rng.normal(size=128).astype(np.float32)
        writer = GGUFWriter(tmp_path / "q.gguf")
        writer.add_tensor("w", data, ggml_type="q4_0")
        writer.write()

        packed, scales = quantize_q4_0(data)
        expected = scales.astype("<f2").tobytes() + np.ascontiguousarray(packed).tobytes()
        assert GGUFReader(tmp_path / "q.gguf").read_tensor_raw("w") == expected

    def test_torch_tensor_accepted(self, tmp_path):
        data = torch.arange(64, dtype=torch.float32).reshape(2, 32)
        writer = GGUFWriter(tmp_path / "t.gguf")
        writer.add_tensor("t", data, ggml_type="f16")
        writer.write()
        assert np.array_equal(GGUFReader(tmp_path / "t.gguf").read_tensor("t"), data.numpy())

    def test_multiple_tensors_order_preserved(self, tmp_path):
        writer = GGUFWriter(tmp_path / "multi.gguf")
        writer.add_tensor("a", np.zeros(32, dtype=np.float32), ggml_type="f16")
        writer.add_tensor("b", np.zeros(64, dtype=np.float32), ggml_type="q8_0")
        writer.add_tensor("c", np.zeros((2, 32), dtype=np.float32), ggml_type="f32")
        writer.write()
        assert list(GGUFReader(tmp_path / "multi.gguf").tensors) == ["a", "b", "c"]

    def test_duplicate_name_raises(self):
        writer = GGUFWriter("x.gguf")
        writer.add_tensor("w", np.zeros(32, dtype=np.float32), ggml_type="f16")
        with pytest.raises(ValueError, match="duplicate tensor name"):
            writer.add_tensor("w", np.zeros(32, dtype=np.float32), ggml_type="f16")

    def test_non_float_input_raises(self):
        writer = GGUFWriter("x.gguf")
        with pytest.raises(ValueError, match="only supports floating-point"):
            writer.add_tensor("w", np.zeros(32, dtype=np.int64), ggml_type="f16")

    def test_unsupported_type_raises(self):
        writer = GGUFWriter("x.gguf")
        with pytest.raises(GGUFError, match="unsupported GGML tensor type"):
            writer.add_tensor("w", np.zeros(32, dtype=np.float32), ggml_type="q4_1")

    def test_quantized_last_dim_must_be_multiple_of_32(self):
        writer = GGUFWriter("x.gguf")
        with pytest.raises(ValueError, match="multiple of 32"):
            writer.add_tensor("w", np.zeros((3, 10), dtype=np.float32), ggml_type="q4_0")

    def test_missing_tensor_raises_keyerror(self, tmp_path):
        writer = GGUFWriter(tmp_path / "x.gguf")
        writer.add_tensor("w", np.zeros(32, dtype=np.float32), ggml_type="f16")
        writer.write()
        with pytest.raises(KeyError, match="no tensor named 'nope'"):
            GGUFReader(tmp_path / "x.gguf").read_tensor("nope")


class TestByteLayout:
    """On-disk structure: reversed dims, 32-byte alignment, correct offsets."""

    def test_dims_stored_reversed(self, tmp_path):
        writer = GGUFWriter(tmp_path / "dims.gguf")
        writer.add_tensor("w", np.zeros((3, 64), dtype=np.float32), ggml_type="f32")
        writer.write()

        data = (tmp_path / "dims.gguf").read_bytes()
        pos = GGUF_HEADER_SIZE  # kv_count == 0, straight to tensor info
        (name_len,) = struct.unpack_from("<Q", data, pos)
        pos += 8 + name_len
        (n_dims,) = struct.unpack_from("<I", data, pos)
        pos += 4
        dims = struct.unpack_from(f"<{n_dims}Q", data, pos)
        assert dims == (64, 3)

        reader = GGUFReader(tmp_path / "dims.gguf")
        assert reader.tensors["w"].shape == (3, 64)

    def test_tensor_offsets_are_aligned_and_padded(self, tmp_path):
        writer = GGUFWriter(tmp_path / "align.gguf")
        # payloads: 34 bytes (Q8_0 block) and 32 bytes (F16).
        writer.add_tensor("a", np.zeros(64, dtype=np.float32), ggml_type="q8_0")
        writer.add_tensor("b", np.zeros(16, dtype=np.float32), ggml_type="f16")
        path = writer.write()

        reader = GGUFReader(path)
        info_a = reader.tensors["a"]
        info_b = reader.tensors["b"]
        assert info_a.offset % GGUF_DEFAULT_ALIGNMENT == 0
        assert info_b.offset % GGUF_DEFAULT_ALIGNMENT == 0
        assert info_b.offset == info_a.offset + align_up(info_a.data_size, GGUF_DEFAULT_ALIGNMENT)
        # The file ends exactly after the last padded payload.
        assert path.stat().st_size == info_b.offset + align_up(info_b.data_size, GGUF_DEFAULT_ALIGNMENT)

    def test_data_section_start_is_aligned(self, tmp_path):
        writer = GGUFWriter(tmp_path / "s.gguf")
        writer.add_metadata("long_key_name", "some value that makes the head unaligned")
        writer.add_tensor("w", np.zeros(64, dtype=np.float32), ggml_type="q8_0")
        writer.write()
        reader = GGUFReader(tmp_path / "s.gguf")
        assert reader.tensors["w"].offset % GGUF_DEFAULT_ALIGNMENT == 0


class TestCorruptFiles:
    """Every malformed-file class raises GGUFError with a useful message."""

    def test_wrong_magic(self, tmp_path):
        data = _assemble(_header(magic=0xDEADBEEF) + _tensor_info())
        path = tmp_path / "bad_magic.gguf"
        path.write_bytes(data)
        with pytest.raises(GGUFError, match="bad magic"):
            GGUFReader(path)

    def test_file_too_small(self, tmp_path):
        path = tmp_path / "tiny.gguf"
        path.write_bytes(b"\x00" * 10)
        with pytest.raises(GGUFError, match="too small"):
            GGUFReader(path)

    @pytest.mark.parametrize("version", [0, 99])
    def test_unsupported_version(self, tmp_path, version: int):
        data = _assemble(_header(version=version) + _tensor_info())
        path = tmp_path / "ver.gguf"
        path.write_bytes(data)
        with pytest.raises(GGUFError, match="unsupported GGUF version"):
            GGUFReader(path)

    def test_truncated_metadata_string(self, tmp_path):
        head = _header(tensors=0, kvs=1) + struct.pack("<Q", 100) + b"ab"
        path = tmp_path / "trunc_meta.gguf"
        path.write_bytes(_assemble(head))
        with pytest.raises(GGUFError, match="truncated GGUF string"):
            GGUFReader(path)

    def test_unknown_metadata_value_type(self, tmp_path):
        _, payload = encode_value(1)
        head = _header(tensors=0, kvs=1) + struct.pack("<Q", 1) + b"k" + struct.pack("<I", 99) + payload
        path = tmp_path / "bad_vtype.gguf"
        path.write_bytes(_assemble(head))
        with pytest.raises(GGUFError, match="unknown GGUF value type code 99"):
            GGUFReader(path)

    def test_truncated_tensor_info(self, tmp_path):
        path = tmp_path / "trunc_info.gguf"
        path.write_bytes(_header() + struct.pack("<Q", 1) + b"w")
        with pytest.raises(GGUFError, match="truncated dimension count"):
            GGUFReader(path)

    def test_truncated_tensor_dims(self, tmp_path):
        path = tmp_path / "trunc_dims.gguf"
        path.write_bytes(_header() + _tensor_info(n_dims=2, dims=()))  # dims array missing
        with pytest.raises(GGUFError, match="truncated dimensions"):
            GGUFReader(path)

    def test_implausible_rank(self, tmp_path):
        head = _header() + _tensor_info(n_dims=100)
        path = tmp_path / "rank.gguf"
        path.write_bytes(_assemble(head))
        with pytest.raises(GGUFError, match="implausible rank"):
            GGUFReader(path)

    def test_unknown_tensor_type_code(self, tmp_path):
        head = _header() + _tensor_info(type_code=77)
        path = tmp_path / "bad_tt.gguf"
        path.write_bytes(_assemble(head))
        with pytest.raises(GGUFError, match="unknown GGML type code 77"):
            GGUFReader(path)

    def test_unsupported_tensor_type(self, tmp_path):
        head = _header() + _tensor_info(type_code=GGMLQuantizationType.Q4_1)
        path = tmp_path / "unsup_tt.gguf"
        path.write_bytes(_assemble(head))
        with pytest.raises(GGUFError, match="unsupported GGML type"):
            GGUFReader(path)

    def test_offset_before_data_section(self, tmp_path):
        head = _header() + _tensor_info(offset=0)
        path = tmp_path / "early.gguf"
        path.write_bytes(_assemble(head, payload=b"\x00" * 34))
        with pytest.raises(GGUFError, match="precedes the data section"):
            GGUFReader(path)

    def test_offset_beyond_file_size(self, tmp_path):
        head = _header() + _tensor_info(offset=1_000_000)
        path = tmp_path / "late.gguf"
        path.write_bytes(_assemble(head, payload=b"\x00" * 34))
        with pytest.raises(GGUFError, match="exceeds file size"):
            GGUFReader(path)
