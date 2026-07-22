"""Tests for LayerQuantPolicy + resolve_layer_policies.

LayerQuantPolicy is the atomic per-layer quantization override primitive
(algorithm-agnostic). resolve_layer_policies is the generic helper that
turns a tuple of policies into a dict[layer_name, effective_config].

These tests cover:
- T1: LayerQuantPolicy dataclass construction & validation (8 tests)
- T2: resolve_layer_policies helper (12 tests)

For LayerQuantPolicy and resolve_layer_policies.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

# =============================================================================
# Level 1: LayerQuantPolicy dataclass (8 tests)
# =============================================================================


def test_policy_construction_minimal():
    """Minimal policy: only target_modules, all override fields None (= inherit)."""
    from llm.quantization._policy import LayerQuantPolicy

    p = LayerQuantPolicy(target_modules=("fc1",))
    assert p.target_modules == ("fc1",)
    assert p.bits is None
    assert p.group_size is None
    assert p.sym is None
    assert p.act_order is None


def test_policy_construction_all_fields():
    """All override fields set explicitly (no inheritance)."""
    from llm.quantization._policy import LayerQuantPolicy

    p = LayerQuantPolicy(
        target_modules=("fc1",),
        bits=8,
        group_size=-1,
        sym=False,
        act_order=True,
    )
    assert p.bits == 8
    assert p.group_size == -1
    assert p.sym is False
    assert p.act_order is True


def test_policy_immutable():
    """frozen dataclass: field assignment raises FrozenInstanceError."""
    from llm.quantization._policy import LayerQuantPolicy

    p = LayerQuantPolicy(target_modules=("fc1",), bits=8)
    with pytest.raises((AttributeError, Exception)) as exc_info:
        p.bits = 4  # type: ignore[misc]
    # dataclasses.FrozenInstanceError is a subclass of AttributeError
    assert "frozen" in str(exc_info.value).lower() or isinstance(exc_info.value, AttributeError)


def test_policy_empty_target_modules_raises():
    """E1: empty target_modules raises ValueError at construction."""
    from llm.quantization._policy import LayerQuantPolicy

    with pytest.raises(ValueError, match="cannot be empty"):
        LayerQuantPolicy(target_modules=())


def test_policy_duplicate_targets_raises():
    """E2: duplicate targets within one policy raises ValueError."""
    from llm.quantization._policy import LayerQuantPolicy

    with pytest.raises(ValueError, match=r"duplicates.*\['fc1'\]"):
        LayerQuantPolicy(target_modules=("fc1", "fc1"))


def test_policy_invalid_bits_raises():
    """E3/E4: bits not in {4, 8} raises ValueError."""
    from llm.quantization._policy import LayerQuantPolicy

    for bad_bits in (3, 16, -1, 0, 5, 32):
        with pytest.raises(ValueError, match=r"bits must be 4, 8, or None"):
            LayerQuantPolicy(target_modules=("fc1",), bits=bad_bits)


def test_policy_invalid_group_size_raises():
    """E5/E6/E7: group_size must be None, -1, or positive int."""
    from llm.quantization._policy import LayerQuantPolicy

    # 0 or negative (except -1) → ValueError
    for bad_gs in (0, -128, -2):
        with pytest.raises(ValueError, match=r"group_size must be -1 .* or positive"):
            LayerQuantPolicy(target_modules=("fc1",), group_size=bad_gs)

    # Non-int (str) → ValueError
    with pytest.raises(ValueError, match=r"group_size must be int or None"):
        LayerQuantPolicy(target_modules=("fc1",), group_size="128")  # type: ignore[arg-type]

    # None and -1 are legal
    LayerQuantPolicy(target_modules=("fc1",), group_size=None)
    LayerQuantPolicy(target_modules=("fc1",), group_size=-1)
    LayerQuantPolicy(target_modules=("fc1",), group_size=128)


def test_policy_invalid_bool_fields_raises():
    """sym / act_order must be bool or None (not truthy strings or ints)."""
    from llm.quantization._policy import LayerQuantPolicy

    # dataclass type-checking already rejects non-bool non-None for bool fields
    with pytest.raises((TypeError, ValueError)):
        LayerQuantPolicy(target_modules=("fc1",), sym="yes")  # type: ignore[arg-type]

    with pytest.raises((TypeError, ValueError)):
        LayerQuantPolicy(target_modules=("fc1",), act_order=1)  # type: ignore[arg-type]

    # None and proper bool are legal
    LayerQuantPolicy(target_modules=("fc1",), sym=None, act_order=None)
    LayerQuantPolicy(target_modules=("fc1",), sym=True, act_order=False)


# =============================================================================
# Level 2: resolve_layer_policies helper (12 tests)
# =============================================================================


def _make_base_config(bits=4, group_size=128, sym=True, act_order=False):
    """Helper: build a minimal base dataclass with the four override fields."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Base:
        bits: int = 4
        group_size: int = 128
        sym: bool = True
        act_order: bool = False

    return Base(bits=bits, group_size=group_size, sym=sym, act_order=act_order)


def test_resolve_empty_policies_returns_empty_dict():
    """T2.1: no policies → empty dict (caller falls back to base config)."""
    from llm.quantization._policy import resolve_layer_policies

    base = _make_base_config()
    result = resolve_layer_policies((), {"fc1", "fc2"}, base)
    assert result == {}


def test_resolve_single_policy_no_overrides():
    """T2.2: policy with all-None overrides → effective ≡ base (no-op dispatch)."""
    from llm.quantization._policy import LayerQuantPolicy, resolve_layer_policies

    base = _make_base_config(bits=8, group_size=-1)
    policy = LayerQuantPolicy(target_modules=("fc1",))  # all None
    result = resolve_layer_policies((policy,), {"fc1"}, base)

    assert set(result.keys()) == {"fc1"}
    effective = result["fc1"]
    assert effective.bits == base.bits
    assert effective.group_size == base.group_size
    assert effective.sym == base.sym
    assert effective.act_order == base.act_order


def test_resolve_single_policy_bits_override():
    """T2.3: bits override applies, other fields inherit."""
    from llm.quantization._policy import LayerQuantPolicy, resolve_layer_policies

    base = _make_base_config(bits=4, group_size=128)
    policy = LayerQuantPolicy(target_modules=("fc1",), bits=8)
    result = resolve_layer_policies((policy,), {"fc1"}, base)

    effective = result["fc1"]
    assert effective.bits == 8
    assert effective.group_size == 128  # inherited


def test_resolve_single_policy_group_size_override():
    """T2.4: group_size override (per-channel -1) applies, bits inherits."""
    from llm.quantization._policy import LayerQuantPolicy, resolve_layer_policies

    base = _make_base_config(bits=4, group_size=128)
    policy = LayerQuantPolicy(target_modules=("fc1",), group_size=-1)
    result = resolve_layer_policies((policy,), {"fc1"}, base)

    effective = result["fc1"]
    assert effective.bits == 4  # inherited
    assert effective.group_size == -1


def test_resolve_single_policy_sym_override():
    """T2.5: sym=False override applies (asymmetric quantization)."""
    from llm.quantization._policy import LayerQuantPolicy, resolve_layer_policies

    base = _make_base_config(sym=True)
    policy = LayerQuantPolicy(target_modules=("fc1",), sym=False)
    result = resolve_layer_policies((policy,), {"fc1"}, base)

    assert result["fc1"].sym is False


def test_resolve_single_policy_act_order_override():
    """T2.6: act_order=True override applies."""
    from llm.quantization._policy import LayerQuantPolicy, resolve_layer_policies

    base = _make_base_config(act_order=False)
    policy = LayerQuantPolicy(target_modules=("fc1",), act_order=True)
    result = resolve_layer_policies((policy,), {"fc1"}, base)

    assert result["fc1"].act_order is True


def test_resolve_multi_policy_dispatch():
    """T2.7: two policies → two effective configs in the map."""
    from llm.quantization._policy import LayerQuantPolicy, resolve_layer_policies

    base = _make_base_config(bits=4, group_size=128)
    p1 = LayerQuantPolicy(target_modules=("fc1",), bits=8, group_size=-1)
    p2 = LayerQuantPolicy(target_modules=("fc2",), bits=4, act_order=True)
    result = resolve_layer_policies((p1, p2), {"fc1", "fc2"}, base)

    assert set(result.keys()) == {"fc1", "fc2"}
    assert result["fc1"].bits == 8
    assert result["fc1"].group_size == -1
    assert result["fc2"].bits == 4
    assert result["fc2"].act_order is True


def test_resolve_unknown_target_raises():
    """E9: policy references a layer not in available_names."""
    from llm.quantization._policy import LayerQuantPolicy, resolve_layer_policies

    base = _make_base_config()
    policy = LayerQuantPolicy(target_modules=("nonexistent",), bits=8)

    with pytest.raises(ValueError, match=r"target_modules.*not found"):
        resolve_layer_policies((policy,), {"fc1", "fc2"}, base)


def test_resolve_cross_policy_overlap_raises():
    """E10: same layer appears in two policies → ValueError."""
    from llm.quantization._policy import LayerQuantPolicy, resolve_layer_policies

    base = _make_base_config()
    p1 = LayerQuantPolicy(target_modules=("fc1",), bits=8)
    p2 = LayerQuantPolicy(target_modules=("fc1",), bits=4)

    with pytest.raises(ValueError, match=r"overlap detected.*\['fc1'\]"):
        resolve_layer_policies((p1, p2), {"fc1", "fc2"}, base)


def test_resolve_inherits_base_for_unmentioned_layers():
    """T2.10: effective_map only covers policy-targeted layers; caller falls back."""
    from llm.quantization._policy import LayerQuantPolicy, resolve_layer_policies

    base = _make_base_config()
    policy = LayerQuantPolicy(target_modules=("fc1",), bits=8)
    result = resolve_layer_policies((policy,), {"fc1", "fc2"}, base)

    # fc1 in map, fc2 NOT in map (caller does result.get("fc2", base))
    assert "fc1" in result
    assert "fc2" not in result


def test_resolve_strips_layer_policies_in_effective():
    """T2.11: effective configs have empty layer_policies (avoid recursion)."""
    from llm.quantization._policy import LayerQuantPolicy, resolve_layer_policies

    # Use a base dataclass that itself has a layer_policies field (like GPTQConfig)
    @dataclass(frozen=True)
    class ConfigWithPolicies:
        bits: int = 4
        group_size: int = 128
        sym: bool = True
        act_order: bool = False
        layer_policies: tuple = ()  # simulate the recursion-vector field

    base = ConfigWithPolicies(layer_policies=(LayerQuantPolicy(target_modules=("fc1",), bits=8),))
    policy = LayerQuantPolicy(target_modules=("fc1",), bits=8)
    result = resolve_layer_policies((policy,), {"fc1"}, base)

    effective = result["fc1"]
    assert effective.layer_policies == ()  # recursion prevented


def test_resolve_generic_over_arbitrary_dataclass():
    """T2.12: helper works for arbitrary dataclass with the four override fields.

    Demonstrates algorithm-agnosticism: future AWQ / SmoothQuant configs can
    reuse this helper without modification.
    """
    from llm.quantization._policy import LayerQuantPolicy, resolve_layer_policies

    @dataclass(frozen=True)
    class FutureAlgoConfig:
        bits: int = 4
        group_size: int = 128
        sym: bool = True
        act_order: bool = False
        # extra algorithm-specific field
        alpha: float = 0.5

    base = FutureAlgoConfig(alpha=0.5)
    policy = LayerQuantPolicy(target_modules=("fc1",), bits=8)
    result = resolve_layer_policies((policy,), {"fc1"}, base)

    effective = result["fc1"]
    assert effective.bits == 8
    # extra field preserved (only override fields get replaced)
    assert effective.alpha == 0.5


# =============================================================================
# Markers
# =============================================================================


pytestmark = pytest.mark.quick
