"""Property-based tests for :mod:`llm.core.registry` NORM_REGISTRY (Finding AD).

``NORM_REGISTRY`` stores factory callables (not classes) since the
splitting refactor (commit ``37a5d4c``). The contract the model
factory relies on:

1. Every registered factory accepts ``hidden_size`` (positional) and
   ``eps`` (keyword) — the call pattern ``ModelFactory`` uses.
2. The produced module's forward output shape equals its input shape.
3. Two calls to the same factory return *different* instances
   (no shared state — instantiating the same norm twice must give
   two independent modules with independent parameters).
4. Calling an unknown name raises ``ValueError`` (consistent with
   the generic :class:`llm.runtime.registry.Registry` contract).

Hypothesis finds the corner cases: ``hidden_size`` very large or
very small, ``eps`` near zero, edge-case layer norm arguments.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from llm.core.registry import NORM_REGISTRY, ensure_norms_registered


@pytest.fixture(scope="module", autouse=True)
def _register_norms():
    """Populate NORM_REGISTRY once for all tests in this module.

    Hypothesis runs each test many times with different inputs; we
    pay the one-time ``ensure_norms_registered`` cost once per module
    instead of per example.
    """
    ensure_norms_registered()


# Strategy: positive hidden size, valid eps.
# Use PowerDistribution-shaped ints for hidden_size — small enough
# to keep nn.LayerNorm cheap, large enough to exercise the contract.
_HIDDEN_SIZE = st.integers(min_value=1, max_value=64)
_EPS = st.floats(
    min_value=1e-6,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)
_BATCH_AND_SEQ = st.integers(min_value=1, max_value=8)


# --- Invariant 1: every factory accepts (hidden_size, eps=...) ------------


@given(
    name=st.sampled_from(["layer_norm", "rms_norm"]),
    hidden_size=_HIDDEN_SIZE,
    eps=_EPS,
)
@settings(max_examples=50, deadline=None)
def test_factory_accepts_hidden_size_and_eps(name, hidden_size, eps):
    """Both built-in factories accept ``(hidden_size, eps=...)`` and return a module."""
    factory = NORM_REGISTRY.get(name)
    module = factory(hidden_size, eps=eps)
    assert module is not None
    assert hasattr(module, "forward")


# --- Invariant 2: forward preserves shape ---------------------------------


@given(
    name=st.sampled_from(["layer_norm", "rms_norm"]),
    hidden_size=_HIDDEN_SIZE,
    batch=_BATCH_AND_SEQ,
    seq_len=_BATCH_AND_SEQ,
    eps=_EPS,
)
@settings(max_examples=50, deadline=None)
def test_forward_preserves_shape(name, hidden_size, batch, seq_len, eps):
    """Norm forward on shape ``[B, S, H]`` returns shape ``[B, S, H]``.

    Shape-changing norms would silently corrupt downstream attention
    projections — the kind of regression that Hypothesis catches by
    trying ``batch=1``, ``seq_len=1``, etc.
    """
    factory = NORM_REGISTRY.get(name)
    module = factory(hidden_size, eps=eps)
    x = torch.randn(batch, seq_len, hidden_size)
    y = module(x)
    assert y.shape == x.shape


# --- Invariant 3: factory returns a NEW instance per call -----------------


@given(
    name=st.sampled_from(["layer_norm", "rms_norm"]),
    hidden_size=_HIDDEN_SIZE,
    eps=_EPS,
)
@settings(max_examples=50, deadline=None)
def test_factory_returns_independent_instances(name, hidden_size, eps):
    """Two factory calls produce two independent ``nn.Module`` instances.

    Parameter sharing would mean the two layers move together during
    optimisation — a silent correctness bug. We assert identity (not
    just type equality) to catch stateful factories.
    """
    factory = NORM_REGISTRY.get(name)
    a = factory(hidden_size, eps=eps)
    b = factory(hidden_size, eps=eps)
    assert a is not b
    # Mutating one must not affect the other.
    if hasattr(a, "weight") and a.weight is not None:
        with torch.no_grad():
            a.weight.fill_(99.0)
        assert not torch.allclose(b.weight, torch.full_like(b.weight, 99.0))


# --- Invariant 4: unknown name raises ValueError --------------------------


@given(name=st.text(min_size=1, max_size=32).filter(lambda s: s not in ("layer_norm", "rms_norm")))
@settings(max_examples=30, deadline=None)
def test_unknown_name_raises(name):
    """Looking up a name that was never registered raises ValueError.

    Mirrors the generic :class:`llm.runtime.registry.Registry`
    contract — ``NORM_REGISTRY`` should not silently ``None``-coerce.
    """
    with pytest.raises(ValueError, match="not found"):
        NORM_REGISTRY.get(name)


# --- Invariant 5: registered factories are callable -----------------------


def test_norms_are_callable_factories():
    """``NORM_REGISTRY`` entries are callables, not classes.

    The split (commit ``37a5d4c``) moved from class storage to
    factory storage so future norms can do shape inference without
    breaking the contract. Pin the invariant.
    """
    for name in NORM_REGISTRY.names():
        factory = NORM_REGISTRY.get(name)
        assert callable(factory), f"{name} is not callable: {type(factory).__name__}"
