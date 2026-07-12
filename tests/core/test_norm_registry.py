"""Tests for NORM_REGISTRY wiring."""

import pytest
from torch import nn

from llm.core.registry import NORM_REGISTRY, ensure_norms_registered
from llm.core.rms_norm import RMSNorm


def test_norm_registry_maps_builtin_implementations():
    """NORM_REGISTRY stores factory callables, not classes.

    The factory is invoked with kwargs to produce an nn.Module instance.
    This matches MODEL_REGISTRY (callables) and lets future norms do
    shape-inference without breaking the registry contract.
    """
    ensure_norms_registered()
    layer_norm_factory = NORM_REGISTRY.get("layer_norm")
    rms_norm_factory = NORM_REGISTRY.get("rms_norm")
    assert callable(layer_norm_factory)
    assert callable(rms_norm_factory)
    # Each factory, when invoked, produces the documented module type.
    assert isinstance(layer_norm_factory(normalized_shape=16), nn.LayerNorm)
    assert isinstance(rms_norm_factory(normalized_shape=16), RMSNorm)


def test_norm_registry_unknown_name_raises():
    ensure_norms_registered()
    with pytest.raises(ValueError, match="not found in"):
        NORM_REGISTRY.get("not_a_norm")
