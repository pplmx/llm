"""GGUF export backend: exporter policy, registry wiring, and model round-trips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from llm.export import (
    EXPORT_REGISTRY,
    ensure_exporters_registered,
    export_model,
)
from llm.export.gguf import GGUFReader, build_gguf_exporter, export_to_gguf
from llm.export.gguf.spec import GGMLQuantizationType, GGUFError
from llm.models.decoder import DecoderModel


@pytest.fixture
def small_model() -> DecoderModel:
    """Tiny CPU decoder; every 2D weight's last dim is 32, so it is block-quantizable."""
    return DecoderModel(
        vocab_size=128,
        hidden_size=32,
        num_layers=1,
        num_heads=2,
        max_seq_len=16,
    )


class _IntBufferModel(nn.Module):
    """Model with a non-floating tensor to exercise the v1 dtype guard."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("ids", torch.arange(8, dtype=torch.long))
        self.fc = nn.Linear(8, 32)


class TestExportPolicy:
    """``export_to_gguf`` type selection and metadata."""

    def test_default_is_f16(self, small_model, tmp_path):
        path = export_to_gguf(small_model, tmp_path / "m.gguf")
        assert path.exists()
        assert path.suffix == ".gguf"
        reader = GGUFReader(path)
        assert {info.ggml_type for info in reader.tensors.values()} == {GGMLQuantizationType.F16}

    def test_f32_export(self, small_model, tmp_path):
        export_to_gguf(small_model, tmp_path / "m.gguf", quantize="f32")
        reader = GGUFReader(tmp_path / "m.gguf")
        assert {info.ggml_type for info in reader.tensors.values()} == {GGMLQuantizationType.F32}

    @pytest.mark.parametrize("quant", ["q4_0", "q8_0"])
    def test_block_quantizes_2d_weights_only(self, small_model, tmp_path, quant: str):
        path = export_to_gguf(small_model, tmp_path / "m.gguf", quantize=quant)
        reader = GGUFReader(path)
        expected = GGMLQuantizationType.Q4_0 if quant == "q4_0" else GGMLQuantizationType.Q8_0
        for name, info in reader.tensors.items():
            if len(info.shape) >= 2:
                assert info.ggml_type is expected, name
            else:
                assert info.ggml_type is GGMLQuantizationType.F16, name

    def test_quantized_export_is_smaller_than_f16(self, small_model, tmp_path):
        f16_path = export_to_gguf(small_model, tmp_path / "f16.gguf")
        q4_path = export_to_gguf(small_model, tmp_path / "q4.gguf", quantize="q4_0")
        assert q4_path.stat().st_size < f16_path.stat().st_size

    def test_metadata_defaults(self, small_model, tmp_path):
        export_to_gguf(small_model, tmp_path / "m.gguf", model_name="tiny-decoder")
        metadata = GGUFReader(tmp_path / "m.gguf").metadata
        assert metadata["general.name"] == "tiny-decoder"
        assert metadata["general.architecture"] == "llm"
        assert metadata["general.file_type"] == 1  # MOSTLY_F16
        assert metadata["general.quantization_version"] == 2

    def test_user_metadata_overrides_defaults(self, small_model, tmp_path):
        export_to_gguf(
            small_model,
            tmp_path / "m.gguf",
            model_name="ignored",
            metadata={"general.name": "mine", "custom.key": [1, 2]},
        )
        metadata = GGUFReader(tmp_path / "m.gguf").metadata
        assert metadata["general.name"] == "mine"
        assert metadata["custom.key"] == [1, 2]

    def test_unknown_quantize_raises(self, small_model, tmp_path):
        with pytest.raises(ValueError, match="quantize must be one of"):
            export_to_gguf(small_model, tmp_path / "m.gguf", quantize="q4_1")

    def test_reader_only_quantize_enum_raises_cleanly(self, small_model, tmp_path):
        """A reader-supported-but-not-exportable type must raise GGUFError,
        not crash with a KeyError in the file_type mapping (round-75 review
        HIGH regression)."""
        # Q4_1 / Q5_0 are still reader-only (all five K-quants are exportable
        # since their write-path milestones).
        for reader_only in (GGMLQuantizationType.Q4_1, GGMLQuantizationType.Q5_0):
            with pytest.raises(GGUFError, match="reader-supported but not exportable"):
                export_to_gguf(small_model, tmp_path / "m.gguf", quantize=reader_only)

    def test_non_float_tensor_raises(self, tmp_path):
        model = _IntBufferModel()
        with pytest.raises(NotImplementedError, match="only supports floating-point"):
            export_to_gguf(model, tmp_path / "m.gguf")

    def test_parent_dirs_created(self, small_model, tmp_path):
        path = export_to_gguf(small_model, tmp_path / "nested" / "dir" / "m.gguf")
        assert path.exists()


class TestModelRoundTrip:
    """Tensors written by the exporter read back with matching names/shapes/values."""

    @pytest.mark.parametrize("quant", [None, "f32", "q4_0", "q8_0"])
    def test_names_and_shapes_match_state_dict(self, small_model, tmp_path, quant):
        path = export_to_gguf(small_model, tmp_path / "m.gguf", quantize=quant)
        state = small_model.state_dict()
        reader = GGUFReader(path)
        assert set(reader.tensors) == set(state)
        for name, info in reader.tensors.items():
            assert info.shape == tuple(state[name].shape)

    def test_f16_values_close_to_original(self, small_model, tmp_path):
        path = export_to_gguf(small_model, tmp_path / "m.gguf")
        state = small_model.state_dict()
        reader = GGUFReader(path)
        for name, expected in state.items():
            np.testing.assert_allclose(
                reader.read_tensor(name),
                expected.numpy(),
                rtol=1e-3,
                atol=1e-3,
            )

    def test_q8_0_weights_stay_close_to_original(self, small_model, tmp_path):
        path = export_to_gguf(small_model, tmp_path / "m.gguf", quantize="q8_0")
        state = small_model.state_dict()
        reader = GGUFReader(path)
        for name, expected in state.items():
            recovered = reader.read_tensor(name)
            if expected.dim() >= 2:
                # Q8_0: quantization half-step (amax/254) plus fp16 scale rounding
                # (amax/2048) bound the reconstruction error.
                max_abs = float(expected.abs().max())
                bound = max_abs / 254.0 + max_abs / 2048.0 + 1e-6
                assert float(np.max(np.abs(recovered - expected.numpy()))) <= bound


class TestRegistryWiring:
    """``gguf`` resolves through EXPORT_REGISTRY like torchscript."""

    def test_gguf_registered_after_bootstrap(self):
        ensure_exporters_registered()
        assert "gguf" in EXPORT_REGISTRY
        assert EXPORT_REGISTRY.get("gguf") is build_gguf_exporter

    def test_all_targets_resolve(self):
        ensure_exporters_registered()
        names = EXPORT_REGISTRY.names()
        assert "onnx" in names
        assert "torchscript" in names
        assert "gguf" in names

    def test_export_model_routes_to_gguf(self, small_model, tmp_path):
        output_path = tmp_path / "via_registry.gguf"
        result = export_model("gguf", small_model, output_path, quantize="q4_0")

        assert isinstance(result, Path)
        assert result == output_path
        assert output_path.exists()
        reader = GGUFReader(output_path)
        assert {info.ggml_type for info in reader.tensors.values()} <= {
            GGMLQuantizationType.Q4_0,
            GGMLQuantizationType.F16,
        }
