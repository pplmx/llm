"""Format-level tests: GGUF constants, header layout, metadata KV encoding, tensor-info math."""

from __future__ import annotations

import struct

import pytest

from llm.export.gguf.metadata import (
    decode_metadata,
    decode_value,
    encode_metadata,
    encode_value,
    value_type,
)
from llm.export.gguf.spec import (
    GGML_BLOCK_SIZE,
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_HEADER_SIZE,
    GGUF_MAGIC,
    GGUF_VERSION,
    GGMLQuantizationType,
    GGUFError,
    GGUFValueType,
    can_quantize_shape,
    parse_ggml_type,
    tensor_data_size,
)


class TestConstants:
    """The format's magic numbers and layout are pinned by the spec."""

    def test_magic_is_ascii_gguf(self):
        assert GGUF_MAGIC == 0x4655_4747
        assert GGUF_MAGIC.to_bytes(4, "little") == b"GGUF"

    def test_version_and_alignment(self):
        assert GGUF_VERSION == 3
        assert GGUF_DEFAULT_ALIGNMENT == 32
        assert GGML_BLOCK_SIZE == 32

    def test_header_struct_size(self):
        # magic u32 + version u32 + tensor count u64 + KV count u64.
        assert struct.calcsize("<IIQQ") == GGUF_HEADER_SIZE == 24


class TestValueTypeCodes:
    """The 13 GGUFValueType codes (spec §Value Types)."""

    @pytest.mark.parametrize(
        ("name", "code"),
        [
            ("UINT8", 0),
            ("INT8", 1),
            ("UINT16", 2),
            ("INT16", 3),
            ("UINT32", 4),
            ("INT32", 5),
            ("FLOAT32", 6),
            ("BOOL", 7),
            ("STRING", 8),
            ("ARRAY", 9),
            ("UINT64", 10),
            ("INT64", 11),
            ("FLOAT64", 12),
        ],
    )
    def test_code(self, name: str, code: int):
        assert GGUFValueType[name].value == code


class TestGGMLTypeCodes:
    """GGML tensor type codes; v1 implements F32/F16/Q4_0/Q8_0."""

    def test_supported_codes_are_stable(self):
        assert GGMLQuantizationType.F32 == 0
        assert GGMLQuantizationType.F16 == 1
        assert GGMLQuantizationType.Q4_0 == 2
        assert GGMLQuantizationType.Q8_0 == 8

    def test_other_well_known_codes(self):
        assert GGMLQuantizationType.Q4_1 == 3
        assert GGMLQuantizationType.Q5_0 == 6
        assert GGMLQuantizationType.Q5_1 == 7
        assert GGMLQuantizationType.Q8_1 == 9
        assert GGMLQuantizationType.Q2_K == 10

    def test_parse_names_are_case_insensitive(self):
        assert parse_ggml_type("q4_0") is GGMLQuantizationType.Q4_0
        assert parse_ggml_type("Q8_0") is GGMLQuantizationType.Q8_0
        assert parse_ggml_type("f16") is GGMLQuantizationType.F16

    def test_parse_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown GGML type"):
            parse_ggml_type("nope")


class TestTensorDataSize:
    """On-disk payload sizes follow the block layouts."""

    def test_raw_types_scale_with_numel(self):
        assert tensor_data_size(GGMLQuantizationType.F32, (3, 64)) == 3 * 64 * 4
        assert tensor_data_size(GGMLQuantizationType.F16, (3, 64)) == 3 * 64 * 2

    def test_q4_0_block_size(self):
        # 18 bytes per 32-element block: fp16 scale + 16 nibble bytes.
        assert tensor_data_size(GGMLQuantizationType.Q4_0, (4, 64)) == (256 // 32) * 18

    def test_q8_0_block_size(self):
        # 34 bytes per 32-element block: fp16 scale + 32 int8 bytes.
        assert tensor_data_size(GGMLQuantizationType.Q8_0, (4, 64)) == (256 // 32) * 34

    def test_quantized_non_multiple_raises(self):
        with pytest.raises(GGUFError, match="multiple of 32"):
            tensor_data_size(GGMLQuantizationType.Q4_0, (3, 10))

    def test_unsupported_type_raises(self):
        with pytest.raises(GGUFError, match="unsupported GGML tensor type"):
            tensor_data_size(GGMLQuantizationType.Q4_1, (3, 64))

    def test_can_quantize_shape(self):
        assert can_quantize_shape((3, 64))
        assert can_quantize_shape((32,))
        assert not can_quantize_shape((3, 10))
        assert not can_quantize_shape((16,))
        assert not can_quantize_shape(())


class TestValueInference:
    """Writer-side Python → GGUFValueType mapping."""

    def test_scalar_mapping(self):
        assert value_type(5) is GGUFValueType.INT64
        assert value_type(1.5) is GGUFValueType.FLOAT32
        assert value_type(True) is GGUFValueType.BOOL
        assert value_type("x") is GGUFValueType.STRING

    def test_array_mapping(self):
        assert value_type([1, 2, 3]) is GGUFValueType.ARRAY
        assert value_type([]) is GGUFValueType.ARRAY
        assert value_type(("a", "b")) is GGUFValueType.ARRAY

    def test_bool_is_not_int(self):
        # bool must not be inferred as INT64 (it subclasses int in Python).
        assert value_type(True) is GGUFValueType.BOOL

    def test_mixed_array_raises(self):
        with pytest.raises(GGUFError, match="homogeneous"):
            value_type([1, "a"])

    def test_nested_array_raises(self):
        with pytest.raises(GGUFError, match="nested arrays"):
            value_type([[1], [2]])

    def test_unencodable_raises(self):
        with pytest.raises(GGUFError, match="cannot encode"):
            value_type(None)
        with pytest.raises(GGUFError, match="cannot encode"):
            value_type({"a": 1})


class TestValueEncodeDecode:
    """Binary encode/decode round-trips and error paths."""

    @pytest.mark.parametrize(
        "value",
        [0, -1, 2**40, True, False, "hello", 1.5, -0.25, [1, 2, 3], ["a", "b"], [], [1.5, 2.5]],
    )
    def test_roundtrip(self, value):
        t, payload = encode_value(value)
        decoded, consumed = decode_value(t, payload)
        assert decoded == value
        assert consumed == len(payload)

    def test_uint8_explicit_decode(self):
        value, consumed = decode_value(GGUFValueType.UINT8, b"\x2a")
        assert value == 42
        assert consumed == 1

    def test_float64_decode(self):
        payload = struct.pack("<d", 3.141592653589793)
        value, consumed = decode_value(GGUFValueType.FLOAT64, payload)
        assert value == 3.141592653589793
        assert consumed == 8

    def test_unknown_type_code_raises(self):
        with pytest.raises(GGUFError, match="unknown GGUF value type code 99"):
            decode_value(99, b"")

    def test_truncated_fixed_value_raises(self):
        with pytest.raises(GGUFError, match="truncated UINT32"):
            decode_value(GGUFValueType.UINT32, b"\x01")

    def test_truncated_string_raises(self):
        with pytest.raises(GGUFError, match="truncated GGUF string"):
            decode_value(GGUFValueType.STRING, b"\x05\x00\x00\x00\x00\x00\x00\x00ab")

    def test_truncated_array_header_raises(self):
        with pytest.raises(GGUFError, match="truncated ARRAY metadata header"):
            decode_value(GGUFValueType.ARRAY, b"\x01\x00\x00\x00")


class TestMetadataBlob:
    """Whole-dict serialization used by the writer/reader."""

    def test_encode_decode_roundtrip(self):
        blob = encode_metadata(
            {
                "general.name": "tiny",
                "general.file_type": 1,
                "general.some_float": 0.5,
                "enabled": True,
                "tags": ["a", "b"],
                "empty": [],
            }
        )
        assert decode_metadata(blob, 6) == {
            "general.name": "tiny",
            "general.file_type": 1,
            "general.some_float": 0.5,
            "enabled": True,
            "tags": ["a", "b"],
            "empty": [],
        }

    def test_empty_metadata(self):
        assert decode_metadata(b"", 0) == {}

    def test_bad_key_raises(self):
        with pytest.raises(ValueError, match="metadata key"):
            encode_metadata({"": 1})

    def test_truncated_blob_raises(self):
        blob = encode_metadata({"k": 1})
        with pytest.raises(GGUFError):
            decode_metadata(blob[: len(blob) - 2], 1)
