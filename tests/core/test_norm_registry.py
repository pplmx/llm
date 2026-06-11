"""Tests for NORM_REGISTRY wiring."""

from torch import nn

from llm.core.registry import NORM_REGISTRY, ensure_norms_registered
from llm.core.rms_norm import RMSNorm


def test_norm_registry_contains_builtins():
    ensure_norms_registered()
    assert "layer_norm" in NORM_REGISTRY
    assert "rms_norm" in NORM_REGISTRY
    assert NORM_REGISTRY.get("layer_norm") is nn.LayerNorm
    assert NORM_REGISTRY.get("rms_norm") is RMSNorm
