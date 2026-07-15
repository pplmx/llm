"""Tests for the export registry (Finding BH)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm.export import (
    EXPORT_REGISTRY,
    ExportBackendFactory,
    build_onnx_exporter,
    ensure_exporters_registered,
    export_model,
)
from llm.models.decoder import DecoderModel


@pytest.fixture
def small_model() -> DecoderModel:
    """Tiny CPU model for export smoke tests.

    ``vocab_size=128`` matches the upper bound of the random dummy
    input that :func:`llm.export.onnx.export_to_onnx` generates
    internally (``torch.randint(0, 100, ...)``); values up to 99 are
    sampled, so any model with ``vocab_size < 100`` will hit an
    embedding out-of-range. Mirrors the fixture in
    ``tests/export/test_export.py``.
    """
    return DecoderModel(
        vocab_size=128,
        hidden_size=32,
        num_layers=1,
        num_heads=2,
        max_seq_len=16,
    )


class TestRegistryShape:
    """The registry mirrors ``generation/registry.py``."""

    def test_onnx_is_built_in(self):
        """The ``onnx`` target is registered by ``ensure_exporters_registered``."""
        ensure_exporters_registered()
        assert "onnx" in EXPORT_REGISTRY
        assert EXPORT_REGISTRY.get("onnx") is build_onnx_exporter

    def test_names_include_onnx(self):
        ensure_exporters_registered()
        assert "onnx" in EXPORT_REGISTRY.names()

    def test_registry_type_alias_exported(self):
        # The type alias is part of the public surface so plugin authors
        # can type-hint their factories without reaching into the
        # registry module directly.
        from llm.export import ExportBackendFactory as Alias

        assert Alias is ExportBackendFactory


class TestEnsureIdempotent:
    """``ensure_exporters_registered`` is safe to call repeatedly."""

    def test_repeat_call_is_noop(self):
        """Calling the bootstrap twice must not attempt to re-register.

        The module-level guard short-circuits the second call so it
        never reaches the ``EXPORT_REGISTRY.register("onnx", ...)``
        line. We verify by checking the registry contents and the
        guard are unchanged across the second call — if the guard
        regressed, the second call would raise ``"already registered"``
        because ``onnx`` is in fact in the registry from the first
        call (or from a previous test that called
        ``ensure_exporters_registered``).
        """
        import llm.export.registry as registry_module

        # Ensure the bootstrap has run at least once so we can
        # observe a no-op second call. If we ran in isolation, the
        # flag would start False and the first call would register;
        # either way, the SECOND call must be a no-op.
        ensure_exporters_registered()
        first_entry = EXPORT_REGISTRY._entries["onnx"]
        assert registry_module._exporters_registered is True

        # The guard must short-circuit this call — if it didn't,
        # ``EXPORT_REGISTRY.register("onnx", ...)`` would raise
        # because onnx is already in the registry.
        ensure_exporters_registered()
        assert EXPORT_REGISTRY._entries["onnx"] is first_entry
        assert registry_module._exporters_registered is True

    def test_unknown_target_raises(self):
        ensure_exporters_registered()
        with pytest.raises(ValueError, match="not found in ExportBackend registry"):
            EXPORT_REGISTRY.get("definitely_not_a_real_target")


class TestExportModelDispatch:
    """``export_model`` resolves and dispatches by name."""

    def test_export_model_routes_to_onnx(self, small_model, tmp_path):
        """``export_model("onnx", ...)`` writes the same artifact as the legacy API."""
        output_path = tmp_path / "via_registry.onnx"
        result = export_model("onnx", small_model, output_path, input_shape=(1, 8))

        assert isinstance(result, Path)
        assert result == output_path
        assert output_path.exists()

    def test_export_model_forwards_kwargs(self, small_model, tmp_path):
        """Target-specific kwargs (``input_shape``, ``opset_version``, ...) pass through."""
        output_path = tmp_path / "kwarg.onnx"
        result = export_model(
            "onnx",
            small_model,
            output_path,
            input_shape=(2, 8),
            opset_version=17,
        )
        assert result.exists()

    def test_export_model_unknown_target_raises(self, small_model, tmp_path):
        with pytest.raises(ValueError, match="not found in ExportBackend"):
            export_model("totally_made_up_target", small_model, tmp_path / "x.onnx")
