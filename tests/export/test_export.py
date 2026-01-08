"""Tests for ONNX export utilities."""

import pytest

from llm.export import export_to_onnx, get_onnx_info, verify_onnx
from llm.models.decoder import DecoderModel


@pytest.fixture
def small_model():
    """Create a small model for testing."""
    return DecoderModel(
        vocab_size=100,
        hidden_size=32,
        num_layers=1,
        num_heads=2,
        max_seq_len=64,
    )


class TestExportToOnnx:
    """Tests for export_to_onnx function."""

    def test_export_creates_file(self, small_model, tmp_path):
        """Test that export creates an ONNX file."""
        output_path = tmp_path / "model.onnx"
        result = export_to_onnx(small_model, output_path)

        assert result.exists()
        assert result.suffix == ".onnx"

    def test_export_custom_input_shape(self, small_model, tmp_path):
        """Test export with custom input shape."""
        output_path = tmp_path / "model.onnx"
        export_to_onnx(small_model, output_path, input_shape=(2, 16))

        assert output_path.exists()

    def test_export_creates_parent_dirs(self, small_model, tmp_path):
        """Test that export creates parent directories."""
        output_path = tmp_path / "nested" / "dir" / "model.onnx"
        export_to_onnx(small_model, output_path)

        assert output_path.exists()


class TestVerifyOnnx:
    """Tests for verify_onnx function."""

    def test_verify_without_model(self, small_model, tmp_path):
        """Test verification without PyTorch comparison."""
        onnx = pytest.importorskip("onnxruntime")  # noqa: F841

        output_path = tmp_path / "model.onnx"
        export_to_onnx(small_model, output_path)

        result = verify_onnx(output_path)
        assert result is True

    def test_verify_with_model_comparison(self, small_model, tmp_path):
        """Test verification with PyTorch output comparison."""
        onnx = pytest.importorskip("onnxruntime")  # noqa: F841

        output_path = tmp_path / "model.onnx"
        export_to_onnx(small_model, output_path)

        # Move model to CPU for comparison
        small_model.cpu()
        result = verify_onnx(output_path, model=small_model, rtol=1e-2, atol=1e-3)
        assert result is True


class TestGetOnnxInfo:
    """Tests for get_onnx_info function."""

    def test_get_info(self, small_model, tmp_path):
        """Test getting ONNX model info."""
        onnx = pytest.importorskip("onnx")  # noqa: F841

        output_path = tmp_path / "model.onnx"
        export_to_onnx(small_model, output_path)

        info = get_onnx_info(output_path)

        assert "opset_version" in info
        assert "inputs" in info
        assert "outputs" in info
        assert "file_size_mb" in info
        assert info["opset_version"] >= 17
        assert len(info["inputs"]) == 1
        assert info["inputs"][0]["name"] == "input_ids"
