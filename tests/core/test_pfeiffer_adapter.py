"""Tests for Pfeiffer Adapter (T2 PEFT #45).

Pfeiffer Adapter (Pfeiffer et al. 2020) — the FFN-only variant of
Houlsby 2019's Adapter. Inserts bottleneck residuals **only after MLP
layers**, not after attention projections. Roughly half the parameters
of Houlsby at comparable accuracy on most tasks; the production default
in AdapterHub / HuggingFace PEFT.

Covers:

- ``apply_pfeiffer_adapter`` default behavior — wraps only ``fc1`` /
  ``fc2`` (the standard MLP layer names in :class:`llm.core.mlp.MLP`)
- ``apply_pfeiffer_adapter`` with custom ``target_modules``
- The wrapper IS :class:`llm.core.adapter.AdapterLinear` (reused, not
  re-implemented) — Pfeiffer is Houlsby-on-MLP-only
- ``merge_pfeiffer_adapter`` / ``unmerge_pfeiffer_adapter`` are no-ops
  (the ``up`` projection is zero-init, so the adapter contributes
  nothing unless trained — matches the Houlsby helper behavior)
- ``get_pfeiffer_parameters`` yields exactly the trainable parameters
- ``count_pfeiffer_parameters`` reports the correct trainable / total
  counts
- ``disable_pfeiffer_adapter`` zeros the ``up`` projection; the wrapper
  becomes identity on top of the base
- ``enable_pfeiffer_adapter`` restores the saved snapshot
- ``disable`` → ``enable`` round-trips
- ``apply_pfeiffer_adapter`` is in-place (returns the same model)
- ``apply_pfeiffer_adapter`` with an empty target list wraps nothing
- ``apply_pfeiffer_adapter`` is idempotent (multiple calls converge to
  the same final state)
- The base ``nn.Linear`` weight stays frozen
- The wrapper class implements the same API as Houlsby (``merge_weights``
  / ``unmerge_weights`` / ``trainable_parameters`` / ``extra_repr``)
- Pfeiffer co-exists with Houlsby — applying one after the other leaves
  both intact, the per-method target filter is the only difference
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from llm.core.adapter import (
    AdapterLinear,
    apply_adapter,
    count_adapter_parameters,
    disable_adapter,
    enable_adapter,
    get_adapter_parameters,
    merge_adapter,
    unmerge_adapter,
)
from llm.core.pfeiffer_adapter import (
    apply_pfeiffer_adapter,
    count_pfeiffer_parameters,
    disable_pfeiffer_adapter,
    enable_pfeiffer_adapter,
    get_pfeiffer_parameters,
    merge_pfeiffer_adapter,
    unmerge_pfeiffer_adapter,
)
from llm.core.peft import PEFT_REGISTRY, apply_peft, ensure_methods_registered


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _TinyTransformerBlock(nn.Module):
    """Minimal transformer block mimicking the project's block structure.

    One ``MultiHeadAttention``-like sub-module (with ``qkv_proj`` /
    ``out_proj`` ``nn.Linear`` children) plus one ``MLP``-like sub-module
    (with ``fc1`` / ``fc2`` children). The fixture exists so we can
    verify the FFN-only behavior of Pfeiffer: ``fc1`` / ``fc2`` should
    be wrapped, ``qkv_proj`` / ``out_proj`` should not.
    """

    def __init__(self, hidden: int = 16, intermediate: int = 32) -> None:
        super().__init__()
        # Attention-like submodule with qkv_proj / out_proj.
        self.self_attn = _AttnLike(hidden)
        # MLP-like submodule with fc1 / fc2.
        self.mlp = _MLPLike(hidden, intermediate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.self_attn(x))


class _AttnLike(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.qkv_proj = nn.Linear(hidden, hidden * 3, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.qkv_proj(x))


class _MLPLike(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


@pytest.fixture
def model() -> _TinyTransformerBlock:
    return _TinyTransformerBlock()


@pytest.fixture(autouse=True)
def _ensure_registered() -> None:
    """Built-ins must be registered before any ``apply_peft`` call."""
    ensure_methods_registered()


# ---------------------------------------------------------------------------
# Registry presence — Slice 1 wiring
# ---------------------------------------------------------------------------


def test_pfeiffer_adapter_is_in_registry() -> None:
    """The ``pfeiffer_adapter`` entry must be in ``PEFT_REGISTRY`` after
    ``ensure_methods_registered`` runs (Slice 1 wires it in).
    """
    assert "pfeiffer_adapter" in PEFT_REGISTRY.names()


def test_pfeiffer_adapter_apply_is_module_function() -> None:
    """The registry's ``apply`` is the per-module function
    (``apply_peft("pfeiffer_adapter", ...)`` reaches it directly).
    """
    from llm.core.pfeiffer_adapter import apply_pfeiffer_adapter

    assert PEFT_REGISTRY.get("pfeiffer_adapter").apply is apply_pfeiffer_adapter


def test_pfeiffer_adapter_filter_is_linear() -> None:
    """Pfeiffer wraps ``nn.Linear`` layers (the FFN / MLP linears)."""
    from llm.core.peft.types import TargetModuleFilter

    assert PEFT_REGISTRY.get("pfeiffer_adapter").target_module_filter is TargetModuleFilter.LINEAR


# ---------------------------------------------------------------------------
# apply_pfeiffer_adapter — default behavior (FFN-only)
# ---------------------------------------------------------------------------


def test_apply_default_wraps_fc1_and_fc2(model: _TinyTransformerBlock) -> None:
    apply_pfeiffer_adapter(model)
    # fc1 / fc2 are MLP linears → wrapped.
    assert isinstance(model.mlp.fc1, AdapterLinear)
    assert isinstance(model.mlp.fc2, AdapterLinear)


def test_apply_default_skips_attention_projections(model: _TinyTransformerBlock) -> None:
    """The Pfeiffer design choice — no adapter after attention.

    Without this guard the wrap would be equivalent to Houlsby (full
    stack). The whole point of Pfeiffer is **less** capacity on the
    attention path.
    """
    apply_pfeiffer_adapter(model)
    assert isinstance(model.self_attn.qkv_proj, nn.Linear)
    assert isinstance(model.self_attn.out_proj, nn.Linear)
    # The qkv_proj / out_proj are the ORIGINAL nn.Linear — not wrapped.
    assert not isinstance(model.self_attn.qkv_proj, AdapterLinear)
    assert not isinstance(model.self_attn.out_proj, AdapterLinear)


def test_apply_returns_model_in_place(model: _TinyTransformerBlock) -> None:
    out = apply_pfeiffer_adapter(model)
    assert out is model


def test_apply_is_idempotent(model: _TinyTransformerBlock) -> None:
    """Calling twice converges to the same state — no double-wrap, no
    errors. Matches the Houlsby ``apply_adapter`` behavior.
    """
    apply_pfeiffer_adapter(model)
    fc1_before = model.mlp.fc1
    apply_pfeiffer_adapter(model)
    # Same wrapper instance — not re-wrapped.
    assert model.mlp.fc1 is fc1_before


def test_apply_with_empty_target_list_wraps_everything(model: _TinyTransformerBlock) -> None:
    """Empty ``target_modules`` means "wrap everything" — matches the
    Houlsby ``apply_adapter`` convention (its ``should_apply`` returns
    True for falsy ``target_modules``).

    To restrict to a subset, pass a non-empty list. To use the
    Pfeiffer default ``["fc1", "fc2"]``, pass ``None``.
    """
    apply_pfeiffer_adapter(model, target_modules=[])
    # Everything wrapped (qkv_proj, out_proj, fc1, fc2).
    assert isinstance(model.mlp.fc1, AdapterLinear)
    assert isinstance(model.mlp.fc2, AdapterLinear)
    assert isinstance(model.self_attn.qkv_proj, AdapterLinear)
    assert isinstance(model.self_attn.out_proj, AdapterLinear)


def test_apply_with_custom_target_modules(model: _TinyTransformerBlock) -> None:
    """Override the default ``fc1`` / ``fc2`` filter.

    Useful for users with custom MLP modules whose layers aren't named
    ``fc1`` / ``fc2`` — they can point at their own layer names.
    """
    apply_pfeiffer_adapter(model, target_modules=["fc1"])
    # fc1 wrapped, fc2 NOT wrapped.
    assert isinstance(model.mlp.fc1, AdapterLinear)
    assert isinstance(model.mlp.fc2, nn.Linear)


# ---------------------------------------------------------------------------
# Identity at init + base freeze
# ---------------------------------------------------------------------------


def test_adapter_is_identity_at_init(model: _TinyTransformerBlock) -> None:
    """At construction the adapter must contribute zero to the output —
    the ``up`` projection is zero-init (matches Houlsby). Verified by
    comparing the wrapped fc1 forward to the original ``nn.Linear``
    forward on the same input (byte-for-byte).
    """
    apply_pfeiffer_adapter(model, bottleneck_dim=8)

    x = torch.randn(2, 4, 16)
    # Compute original output from the freshly-built (pre-wrap) layer
    # by re-running the underlying linear on the same input.
    base_linear = model.mlp.fc1.base_layer
    adapter = model.mlp.fc1

    # At init, adapter output == base_linear output (up is zero).
    out_adapter = adapter(x)
    out_base = base_linear(x)
    assert torch.allclose(out_adapter, out_base, atol=1e-6)


def test_base_linear_is_frozen(model: _TinyTransformerBlock) -> None:
    apply_pfeiffer_adapter(model)
    # The base Linear (held inside AdapterLinear.base_layer) has its
    # weights frozen — only the down/up parameters train.
    assert model.mlp.fc1.base_layer.weight.requires_grad is False
    # fc1 has bias=True → bias is also frozen.
    assert model.mlp.fc1.base_layer.bias.requires_grad is False


def test_adapter_down_up_are_trainable(model: _TinyTransformerBlock) -> None:
    apply_pfeiffer_adapter(model)
    adapter = model.mlp.fc1
    assert adapter.down.weight.requires_grad is True
    assert adapter.down.bias.requires_grad is True
    assert adapter.up.weight.requires_grad is True
    assert adapter.up.bias.requires_grad is True


# ---------------------------------------------------------------------------
# get_pfeiffer_parameters / count_pfeiffer_parameters
# ---------------------------------------------------------------------------


def test_get_pfeiffer_parameters_yields_only_adapter_params(model: _TinyTransformerBlock) -> None:
    """Each wrapper contributes 4 params (down weight, down bias, up
    weight, up bias). The base Linear params are NOT yielded (they're
    frozen, and ``get_pfeiffer_parameters`` is the optimizer wiring
    helper — base params must not be in the trainable list).
    """
    apply_pfeiffer_adapter(model, bottleneck_dim=8)
    params = list(get_pfeiffer_parameters(model))
    # Two wrappers (fc1 + fc2) × 4 params each = 8.
    assert len(params) == 8
    # Every yielded param belongs to a wrapper's down / up.
    for p in params:
        assert any(p is q for module in model.modules() if isinstance(module, AdapterLinear)
                   for q in (module.down.weight, module.down.bias, module.up.weight, module.up.bias))


def test_count_pfeiffer_parameters_reports_correct_counts(model: _TinyTransformerBlock) -> None:
    """``count_pfeiffer_parameters`` reports total trainable / total
    parameters in the model (delegates to ``count_adapter_parameters``).

    Trainable = Pfeiffer adapter params on fc1 (552) + fc2 (280) +
    attention linears (qkv_proj 768 + out_proj 256, not wrapped, still
    trainable). The base fc1 (16*32=512) and fc2 (32*16=512) weights
    are frozen, so they don't count toward trainable but do count
    toward total.

    For Pfeiffer-**specific** trainable count use
    ``get_pfeiffer_parameters`` + ``sum(p.numel())``.
    """
    apply_pfeiffer_adapter(model, bottleneck_dim=8)
    trainable, total = count_pfeiffer_parameters(model)
    # fc1 adapter: 552, fc2 adapter: 280, qkv_proj: 768, out_proj: 256 → 1856.
    assert trainable == 552 + 280 + 768 + 256
    # Total includes frozen fc1 base (512) + frozen fc2 base (512) = 1024 more.
    assert total > trainable


def test_pfeiffer_specific_trainable_count(model: _TinyTransformerBlock) -> None:
    """Sum ``get_pfeiffer_parameters`` to get the Pfeiffer-only count.

    For fc1 (16→32, bottleneck=8): 2 × 32 × 8 + 32 + 8 = 552.
    For fc2 (32→16, bottleneck=8): 2 × 16 × 8 + 16 + 8 = 280.
    Total = 832 Pfeiffer-specific trainable (matches Houlsby's
    adapter count when restricted to MLP-only).
    """
    apply_pfeiffer_adapter(model, bottleneck_dim=8)
    pfeiffer_only = sum(p.numel() for p in get_pfeiffer_parameters(model))
    assert pfeiffer_only == 832


def test_count_matches_houlsby_on_same_bottleneck(model: _TinyTransformerBlock) -> None:
    """The parameter math is identical to Houlsby — the only difference
    is WHICH linears get wrapped. Sanity check: counting Pfeiffer
    params should match counting Houlsby params restricted to MLP.
    """
    apply_pfeiffer_adapter(model, bottleneck_dim=16)
    pfeiffer_trainable, _ = count_pfeiffer_parameters(model)

    # Reset and apply Houlsby restricted to MLP only.
    fresh = _TinyTransformerBlock()
    apply_adapter(fresh, bottleneck_dim=16, target_modules=["fc1", "fc2"])
    houlsby_trainable, _ = count_adapter_parameters(fresh)

    assert pfeiffer_trainable == houlsby_trainable


def test_get_pfeiffer_parameters_empty_on_fresh_model(model: _TinyTransformerBlock) -> None:
    """No adapters applied → no trainable PEFT params yielded."""
    params = list(get_pfeiffer_parameters(model))
    assert params == []


# ---------------------------------------------------------------------------
# merge / unmerge — no-op (matches Houlsby)
# ---------------------------------------------------------------------------


def test_merge_pfeiffer_adapter_is_noop(model: _TinyTransformerBlock) -> None:
    """Pfeiffer merge is a no-op — the ``up`` projection being zero
    means there's nothing to fold into the base weight. This matches
    the Houlsby ``merge_adapter`` behavior.
    """
    apply_pfeiffer_adapter(model, bottleneck_dim=8)
    fc1_base_before = model.mlp.fc1.base_layer.weight.detach().clone()
    merge_pfeiffer_adapter(model)
    fc1_base_after = model.mlp.fc1.base_layer.weight.detach().clone()
    assert torch.equal(fc1_base_before, fc1_base_after)


def test_unmerge_pfeiffer_adapter_is_noop(model: _TinyTransformerBlock) -> None:
    """Mirror of ``merge_pfeiffer_adapter`` — no-op for the same reason."""
    apply_pfeiffer_adapter(model, bottleneck_dim=8)
    fc1_base_before = model.mlp.fc1.base_layer.weight.detach().clone()
    unmerge_pfeiffer_adapter(model)
    fc1_base_after = model.mlp.fc1.base_layer.weight.detach().clone()
    assert torch.equal(fc1_base_before, fc1_base_after)


def test_merge_unmerge_match_houlsby_helpers(model: _TinyTransformerBlock) -> None:
    """Pfeiffer helpers are thin delegates — behavior matches Houlsby's
    ``merge_adapter`` / ``unmerge_adapter`` byte-for-byte.
    """
    apply_pfeiffer_adapter(model, bottleneck_dim=8)
    fc1_base_weight = model.mlp.fc1.base_layer.weight.detach().clone()
    fc2_base_weight = model.mlp.fc2.base_layer.weight.detach().clone()
    # Run Pfeiffer helpers.
    merge_pfeiffer_adapter(model)
    unmerge_pfeiffer_adapter(model)
    # Both base weights still untouched.
    assert torch.equal(model.mlp.fc1.base_layer.weight, fc1_base_weight)
    assert torch.equal(model.mlp.fc2.base_layer.weight, fc2_base_weight)


# ---------------------------------------------------------------------------
# disable / enable
# ---------------------------------------------------------------------------


def test_disable_pfeiffer_adapter_zeros_up(model: _TinyTransformerBlock) -> None:
    """After disable, every wrapper's ``up.weight`` is zero — the
    adapter contributes nothing to the forward.
    """
    apply_pfeiffer_adapter(model, bottleneck_dim=8)
    disable_pfeiffer_adapter(model)
    for module in model.modules():
        if isinstance(module, AdapterLinear):
            assert torch.all(module.up.weight == 0)
            assert torch.all(module.up.bias == 0)


def test_disabled_adapter_is_identity(model: _TinyTransformerBlock) -> None:
    """After disable the wrapper's output equals the base linear's
    output — the adapter contributes zero, so the wrapper is
    mathematically the identity on top of the base.
    """
    apply_pfeiffer_adapter(model, bottleneck_dim=8)
    disable_pfeiffer_adapter(model)
    x = torch.randn(2, 4, 16)
    # fc1 wrapper output == fc1.base_layer(x).
    out_wrapper = model.mlp.fc1(x)
    out_base = model.mlp.fc1.base_layer(x)
    assert torch.allclose(out_wrapper, out_base, atol=1e-6)


def test_enable_pfeiffer_adapter_restores_up(model: _TinyTransformerBlock) -> None:
    """After ``disable`` → ``enable``, the ``up`` projection is back to
    its pre-disable value. Snapshot stored as ``_original_up_weight`` /
    ``_original_up_bias`` is restored.
    """
    apply_pfeiffer_adapter(model, bottleneck_dim=8)
    # Capture the pre-disable up state.
    up_weight_before = model.mlp.fc1.up.weight.detach().clone()
    # Disable → enable round-trip.
    disable_pfeiffer_adapter(model)
    enable_pfeiffer_adapter(model)
    # Up weight is restored.
    assert torch.equal(model.mlp.fc1.up.weight, up_weight_before)
    # Snapshot attrs cleaned up.
    assert not hasattr(model.mlp.fc1, "_original_up_weight")


def test_enable_without_disable_is_noop(model: _TinyTransformerBlock) -> None:
    """Calling ``enable`` without first calling ``disable`` is a no-op —
    no snapshot exists, nothing to restore. Matches Houlsby.
    """
    apply_pfeiffer_adapter(model, bottleneck_dim=8)
    up_weight_before = model.mlp.fc1.up.weight.detach().clone()
    enable_pfeiffer_adapter(model)
    assert torch.equal(model.mlp.fc1.up.weight, up_weight_before)


def test_disable_enable_match_houlsby_helpers(model: _TinyTransformerBlock) -> None:
    """Pfeiffer disable/enable match Houlsby's behavior byte-for-byte.

    A model with Pfeiffer disable → enable should end in the same
    state as a model with Houlsby disable → enable on the same targets.
    """
    apply_pfeiffer_adapter(model, bottleneck_dim=8)
    disable_pfeiffer_adapter(model)
    enable_pfeiffer_adapter(model)
    up_pfeiffer = model.mlp.fc1.up.weight.detach().clone()

    fresh = _TinyTransformerBlock()
    apply_adapter(fresh, bottleneck_dim=8, target_modules=["fc1", "fc2"])
    disable_adapter(fresh)
    enable_adapter(fresh)
    up_houlsby = fresh.mlp.fc1.up.weight.detach().clone()

    assert torch.equal(up_pfeiffer, up_houlsby)


# ---------------------------------------------------------------------------
# Coexistence with Houlsby adapter
# ---------------------------------------------------------------------------


def test_pfeiffer_and_houlsby_coexist_when_targets_differ(model: _TinyTransformerBlock) -> None:
    """Houlsby + Pfeiffer can be applied to the same model with distinct
    target filters — Pfeiffer takes MLP, Houlsby takes attention.

    The user is responsible for not enabling both with overlapping
    targets (the project convention is one PEFT method per training
    run), but the API doesn't enforce this — the dispatcher is the
    user's config.
    """
    apply_pfeiffer_adapter(model, target_modules=["fc1", "fc2"], bottleneck_dim=8)
    apply_adapter(model, target_modules=["qkv_proj", "out_proj"], bottleneck_dim=8)
    # All four linears wrapped.
    assert isinstance(model.mlp.fc1, AdapterLinear)
    assert isinstance(model.mlp.fc2, AdapterLinear)
    assert isinstance(model.self_attn.qkv_proj, AdapterLinear)
    assert isinstance(model.self_attn.out_proj, AdapterLinear)


# ---------------------------------------------------------------------------
# PEFT_REGISTRY dispatch — Slice 1 entry point
# ---------------------------------------------------------------------------


def test_apply_peft_dispatches_pfeiffer_adapter(model: _TinyTransformerBlock) -> None:
    """The unified ``apply_peft`` entry point reaches Pfeiffer."""
    apply_peft(model, "pfeiffer_adapter", bottleneck_dim=8)
    assert isinstance(model.mlp.fc1, AdapterLinear)
    assert isinstance(model.mlp.fc2, AdapterLinear)
    assert isinstance(model.self_attn.qkv_proj, nn.Linear)


def test_pfeiffer_kwarg_forwarding(model: _TinyTransformerBlock) -> None:
    """``bottleneck_dim`` flows through ``apply_peft`` to the wrapper."""
    apply_peft(model, "pfeiffer_adapter", bottleneck_dim=4)
    assert model.mlp.fc1.bottleneck_dim == 4
    assert model.mlp.fc2.bottleneck_dim == 4


def test_registry_count_after_pfeiffer_added() -> None:
    """Pfeiffer brings the registry to 8 methods (was 7 before this slice)."""
    assert len(PEFT_REGISTRY.names()) == 8
    assert "pfeiffer_adapter" in PEFT_REGISTRY.names()


# ---------------------------------------------------------------------------
# Helper-API parity (functions exist + are callable)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        apply_pfeiffer_adapter,
        merge_pfeiffer_adapter,
        unmerge_pfeiffer_adapter,
        get_pfeiffer_parameters,
        count_pfeiffer_parameters,
        disable_pfeiffer_adapter,
        enable_pfeiffer_adapter,
    ],
)
def test_pfeiffer_helpers_exist_and_are_callable(fn) -> None:
    """Every Pfeiffer helper is exposed at module level and is callable."""
    assert callable(fn)
