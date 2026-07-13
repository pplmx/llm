"""Property-based tests for :mod:`llm.runtime.registry` (Finding AD).

The ``Registry`` class backs every plugin extension point in the
framework (model architectures, attention, MLP, norms, generation
backends). It must hold four invariants under arbitrary input:

1. Registering the same name twice raises (no silent overwrite).
2. ``get`` returns the same instance the caller registered.
3. Iteration via :py:meth:`Registry.names` is sorted (insertion-order
   is irrelevant; stable order is what callers rely on for
   documentation and error messages).
4. An empty registry is iterable and ``__contains__`` is False for any
   unregistered name.

Hypothesis searches the input space these tests would otherwise miss:
empty strings, very long strings, unicode, control characters, etc.
"""

from __future__ import annotations

import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llm.runtime.registry import Registry

# We build the registry inline in each test instead of using a pytest
# fixture: Hypothesis replays examples via @example and during
# shrinking, so the function-scoped fixture check would fire.


# --- Invariant 1: duplicate registration raises ---------------------------


@given(name=st.text(min_size=1, max_size=64), value=st.integers())
@settings(max_examples=50, deadline=None)
def test_registering_same_name_twice_raises(name, value):
    """First registration succeeds; the second raises ValueError.

    Silently overwriting a registered entry would make plugin
    extensions order-dependent and nearly impossible to debug when a
    downstream ``get`` returns the wrong factory.
    """
    registry = Registry("HypothesisTest")
    registry.register(name, value)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(name, value + 1)


# --- Invariant 2: get returns the exact same instance ---------------------


@given(name=st.text(min_size=1, max_size=64))
@settings(max_examples=50, deadline=None)
def test_get_returns_same_instance(name):
    """``get(name)`` returns the exact object passed to ``register``.

    Identity (not equality) is the contract: callers rely on
    registry-stored objects being shared singletons across modules
    (e.g. class objects used as plugin factories).
    """
    registry = Registry("HypothesisTest")
    sentinel = object()
    registry.register(name, sentinel)
    assert registry.get(name) is sentinel


# --- Invariant 3: iteration is sorted (stable order) ----------------------


@given(
    pairs=st.lists(
        st.tuples(
            st.text(min_size=1, max_size=32, alphabet=string.ascii_letters + string.digits),
            st.integers(),
        ),
        min_size=1,
        max_size=30,
    )
)
@settings(max_examples=50, deadline=None)
def test_names_are_sorted(pairs):
    """``Registry.names()`` returns names in ascending lexicographic order.

    Insertion order is deliberately not preserved — stable, sorted
    order is what error messages (``"Available: a, b, c"``) and docs
    rely on. Hypothesis-driven fuzzing exercises duplicate keys too:
    we deduplicate inside the test so the registration contract holds.
    """
    registry = Registry("HypothesisTest")
    seen: set[str] = set()
    for name, value in pairs:
        if name in seen:
            continue
        registry.register(name, value)
        seen.add(name)

    expected = sorted(seen)
    assert registry.names() == expected


# --- Invariant 4: empty registry + contains -------------------------------


@given(name=st.text(min_size=1, max_size=64))
@settings(max_examples=50, deadline=None)
def test_unregistered_name_raises_on_get(name):
    """Looking up any name in a fresh registry raises ValueError.

    Even characters that survive string sanitisation (``\x00`` is
    a real worry on some C extensions) should not accidentally hit
    a registered entry.
    """
    registry = Registry("HypothesisTest")
    with pytest.raises(ValueError, match="not found in"):
        registry.get(name)
    assert name not in registry


def test_empty_registry_iterates_empty():
    """An empty registry exposes an empty (sorted) name list."""
    registry = Registry("Empty")
    assert registry.names() == []


# --- Invariant 5: registration round-trips under unicode/control chars ---


@given(name=st.text(min_size=1, max_size=64))
@settings(max_examples=50, deadline=None)
def test_unicode_and_control_chars_round_trip(name):
    """Any non-empty string can be a registry key and round-trips.

    We don't sanitise keys because plugin names are sourced from
    ``@register_attention("mla")`` decorator arguments, which are
    author-controlled. Round-tripping ensures that the registry does
    not strip or transform keys in unexpected ways.
    """
    registry = Registry("HypothesisTest")
    sentinel = object()
    registry.register(name, sentinel)
    assert registry.get(name) is sentinel
    assert name in registry
