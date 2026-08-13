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


def _fp16_export_loads_and_verifies(tmp_path, *, num_layers: int):
    """fp16 export must produce an onnxruntime-loadable artifact whose
    outputs match the eager model (RIL ISS-067). Returns the ORT session."""
    import onnxruntime as ort
    import torch

    model = DecoderModel(
        vocab_size=64,
        hidden_size=32,
        num_layers=num_layers,
        num_heads=4,
        intermediate_size=64,
        max_seq_len=32,
        dtype=torch.float16,
        device="cpu",
        embedding_dropout_p=0.0,
        attn_dropout_p=0.0,
        mlp_dropout_p=0.0,
    )
    model.eval()
    output_path = tmp_path / f"fp16_{num_layers}l.onnx"
    export_to_onnx(model, output_path, input_shape=(1, 8))

    # Regression: this previously failed at load time with
    # "Type Error: Type parameter (T) of Optype (LayerNormalization) bound
    # to different types (tensor(float) and tensor(float16))".
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    assert session is not None
    return model, session


def test_export_fp16_onnxruntime_loadable(tmp_path):
    """Regression (RIL ISS-067): a fp16 model exports successfully but the
    artifact must be loadable by onnxruntime — it previously was not, because
    TorchScript ONNX fusion mislabeled LayerNormalization input types
    (LayerNorm weight=float16, X=float) and onnxruntime rejected the graph
    at session creation."""
    model, session = _fp16_export_loads_and_verifies(tmp_path, num_layers=2)
    assert model is not None
    assert session is not None


def test_export_fp16_onnx_matches_eager(tmp_path):
    """After the LayerNorm type fix the fp16 ONNX artifact must produce
    outputs close to the eager fp16 model (fp16 tolerance)."""
    import numpy as np
    import torch

    model, session = _fp16_export_loads_and_verifies(tmp_path, num_layers=2)

    torch.manual_seed(7)
    ids = torch.randint(0, model.embedding_layer.token_embeddings.num_embeddings, (1, 8))
    onnx_out = session.run(None, {"input_ids": ids.numpy()})[0]
    with torch.no_grad():
        pt_out = model(ids).float().numpy()

    assert onnx_out.shape == pt_out.shape
    assert np.allclose(onnx_out, pt_out, rtol=1e-2, atol=1e-2), (
        f"fp16 ONNX diverges from eager (max |d|={np.abs(onnx_out - pt_out).max():.5f})"
    )


def test_export_to_onnx_small_vocab_no_crash(tmp_path):
    """Regression (RIL ISS-058): a model with ``vocab_size < 100`` must
    export without crashing on an out-of-range dummy token id.

    The old dummy input was ``torch.randint(0, 100, ...)`` — for a
    vocab_size=16 model the embedding is indexed with ids up to 99 and
    raises ``IndexError: index out of range in self``. The dummy must be
    bounded by the model's real vocab.
    """
    model = DecoderModel(
        vocab_size=16,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        max_seq_len=32,
    )
    output_path = tmp_path / "model.onnx"
    result = export_to_onnx(model, output_path, input_shape=(1, 32))
    assert result.exists()


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
