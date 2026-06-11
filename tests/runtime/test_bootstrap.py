"""Tests for runtime bootstrap via setuptools entry points."""

from llm.runtime.bootstrap import ensure_builtins_registered
from llm.runtime.model_factory import MODEL_REGISTRY


def test_bootstrap_registers_models_from_entry_points():
    ensure_builtins_registered()

    names = MODEL_REGISTRY.names()
    assert "decoder" in names
    assert "regression_mlp" in names
