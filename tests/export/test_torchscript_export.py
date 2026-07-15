"""Tests for the TorchScript export target (Tier 3 #11)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from llm.export import EXPORT_REGISTRY, ensure_exporters_registered, export_model
from llm.export.torchscript import build_torchscript_exporter, export_to_torchscript
from llm.models.decoder import DecoderModel


@pytest.fixture
def small_model() -> DecoderModel:
    """Tiny CPU model for export smoke tests.

    Same shape as :func:`tests.export.test_export.small_model` so
    TorchScript's trace and ONNX's trace can share fixtures without
    drift. ``vocab_size=128`` clears the upper bound of the random
    dummy input used internally; ``max_seq_len=64`` clears the
    default ``input_shape=(1, 32)``.
    """
    return DecoderModel(
        vocab_size=128,
        hidden_size=32,
        num_layers=1,
        num_heads=2,
        max_seq_len=64,
    )


class TestRegistryRegistration:
    """``torchscript`` is registered via the ``llm.export_backends`` entry point."""

    def test_torchscript_resolves_after_bootstrap(self):
        ensure_exporters_registered()
        assert "torchscript" in EXPORT_REGISTRY
        assert EXPORT_REGISTRY.get("torchscript") is build_torchscript_exporter

    def test_both_targets_resolve_after_bootstrap(self):
        """Built-in ``onnx`` and entry-point ``torchscript`` both load."""
        ensure_exporters_registered()
        names = EXPORT_REGISTRY.names()
        assert "onnx" in names
        assert "torchscript" in names


class TestExportToTorchscript:
    """``export_to_torchscript`` writes a loadable TorchScript artifact."""

    def test_export_creates_pt_file(self, small_model, tmp_path):
        output_path = tmp_path / "model.pt"
        result = export_to_torchscript(small_model, output_path)

        assert isinstance(result, Path)
        assert result == output_path
        assert output_path.exists()

    def test_artifact_loads_with_torch_jit_load(self, small_model, tmp_path):
        """The written file is a valid TorchScript archive."""
        output_path = tmp_path / "model.pt"
        export_to_torchscript(small_model, output_path, input_shape=(1, 8))

        scripted = torch.jit.load(str(output_path))
        # Calling the loaded module is the contract for "loadable".
        scripted.eval()
        with torch.no_grad():
            out = scripted(torch.randint(0, 100, (1, 8)))
        assert out.shape == (1, 8, 128)  # (batch, seq, vocab)

    def test_artifact_matches_eager_model(self, small_model, tmp_path):
        """Loaded TorchScript output matches the eager model output (numerical equivalence)."""
        small_model.eval()
        output_path = tmp_path / "model.pt"
        export_to_torchscript(small_model, output_path, input_shape=(1, 8))

        scripted = torch.jit.load(str(output_path))
        scripted.eval()

        # Use a deterministic input for both sides.
        torch.manual_seed(0)
        x = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            eager_out = small_model(x, use_cache=False)
            if isinstance(eager_out, tuple):
                eager_out = eager_out[0]
            scripted_out = scripted(x)

        assert torch.allclose(scripted_out, eager_out, atol=1e-5, rtol=1e-4)

    def test_export_creates_parent_dirs(self, small_model, tmp_path):
        output_path = tmp_path / "nested" / "dir" / "model.pt"
        export_to_torchscript(small_model, output_path)
        assert output_path.exists()

    def test_script_method_works(self, small_model, tmp_path):
        """``method='script'`` path runs without raising on a scriptable model.

        Currently expected to xfail because :class:`DecoderModel`'s
        positional encoding uses constructs TorchScript can't compile
        (``Module 'PositionalEncoding' has no attribute 'pos_embedding'``).
        The trace path is the supported one for the current model;
        script is wired up as a future hook for a smaller / scriptable
        submodule export. See ticket #33 "Out of scope".
        """
        output_path = tmp_path / "scripted.pt"
        try:
            export_to_torchscript(small_model, output_path, method="script", input_shape=(1, 8))
        except RuntimeError as exc:
            if "PositionalEncoding" in str(exc) or "has no attribute" in str(exc):
                pytest.xfail(f"TorchScript scripting not yet supported for DecoderModel: {exc}")
            raise
        assert output_path.exists()
        scripted = torch.jit.load(str(output_path))
        scripted.eval()
        with torch.no_grad():
            out = scripted(torch.randint(0, 100, (1, 8)))
        assert out.shape == (1, 8, 128)


class TestRegistryDispatch:
    """``export_model`` routes the ``torchscript`` target through the registry."""

    def test_export_model_routes_to_torchscript(self, small_model, tmp_path):
        output_path = tmp_path / "via_registry.pt"
        result = export_model("torchscript", small_model, output_path, input_shape=(1, 8))

        assert result == output_path
        assert output_path.exists()

    def test_export_model_unknown_target_raises(self, small_model, tmp_path):
        with pytest.raises(ValueError, match="not found in ExportBackend"):
            export_model("definitely_not_real", small_model, tmp_path / "x.pt")
