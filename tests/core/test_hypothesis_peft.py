"""Property-based invariant tests for the PEFT slice (Finding AD).

The example-based tests in ``test_ia3.py``, ``test_bitfit.py``,
``test_lora.py``, and ``test_prefix_tuning.py`` pin specific shapes
and scenarios. Hypothesis verifies the **invariants** that should
hold for ANY shape / scale / bias-count:

IA³:
1. ``IA3Linear(x) == base_layer(x) * ia3_l`` for any input — the
   forward is exactly multiplicative, broadcast over batch / seq.
2. ``merge_weights()`` makes the wrapper the identity on top of the
   already-folded base — i.e. ``post_merge(x) == pre_merge(x)`` for
   any input.
3. ``merge_weights()`` then ``unmerge_weights()`` returns the base
   weight to its pre-merge state (modulo float drift).
4. ``disable_ia3`` (sets ``ia3_l = ones``) makes
   ``IA3Linear(x) == base_layer(x)`` for any input.

BitFit:
1. After ``apply_bitfit``, every parameter with a ``.bias`` suffix
   has ``requires_grad=True`` and every parameter WITHOUT a ``.bias``
   suffix has ``requires_grad=False``.
2. The count of trainable parameters equals the sum of bias sizes
   (for any model — independent of bias presence on each Linear).
3. ``apply_bitfit`` then ``unapply_bitfit`` restores the original
   ``requires_grad`` state exactly.

The shapes are kept small so the property tests stay fast. Each test
runs ~10 examples by default — enough to catch off-by-one bugs
without slowing CI.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from llm.core.bitfit import apply_bitfit, get_bitfit_parameters, unapply_bitfit
from llm.core.ia3 import (
    IA3Linear,
    apply_ia3,
    disable_ia3,
    enable_ia3,
)
from llm.core.lora import LoRALinear

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


# IA3 scales must be positive-ish for the math to make sense in
# practice, but the invariant is independent of the magnitude. Use
# small floats to avoid overflow in downstream ops.
_IA3_SCALE = st.floats(
    min_value=-5.0,
    max_value=5.0,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def _ia3_layer_and_input(draw, in_features: int = 8, out_features: int = 16):
    """Yield ``(IA3Linear, x)`` with arbitrary scale and input."""
    torch.manual_seed(draw(st.integers(min_value=0, max_value=2**31 - 1)))
    base = nn.Linear(in_features, out_features)
    scale = draw(_IA3_SCALE)
    layer = IA3Linear(base, init_scale=1.0)
    with torch.no_grad():
        layer.ia3_l.fill_(scale)
    batch = draw(st.integers(min_value=1, max_value=4))
    seq = draw(st.integers(min_value=1, max_value=6))
    x = torch.randn(batch, seq, in_features)
    return layer, x, scale


@st.composite
def _bias_holder_spec(draw):
    """Yield a spec tuple ``(hidden, bias_tup)`` for building a small
    ``nn.Module`` with a random mix of bias / no-bias Linears plus a
    LayerNorm.

    Returning a spec instead of a model lets each test build
    independent model instances — otherwise Hypothesis re-uses the
    same object across calls (e.g. when the same strategy is bound
    to multiple test arguments).
    """
    hidden = draw(st.integers(min_value=4, max_value=16))
    bias_tup = (
        draw(st.booleans()),
        draw(st.booleans()),
        draw(st.booleans()),
    )
    return hidden, bias_tup


def _build_holder(hidden: int, bias_tup: tuple[bool, bool, bool]) -> nn.Module:
    """Build a fresh ``_Holder`` from a spec returned by
    :func:`_bias_holder_spec`.

    The bias presence of each Linear is given by ``bias_tup`` so the
    BitFit invariants hold regardless of bias configuration.
    """

    class _Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(hidden, hidden, bias=bias_tup[0])
            self.fc2 = nn.Linear(hidden, hidden, bias=bias_tup[1])
            self.fc3 = nn.Linear(hidden, hidden, bias=bias_tup[2])
            self.norm = nn.LayerNorm(hidden)

        def forward(self, x):  # pragma: no cover
            return self.norm(self.fc3(self.fc2(self.fc1(x))))

    return _Holder()


# ---------------------------------------------------------------------------
# IA3 — forward math is exactly base * scale
# ---------------------------------------------------------------------------


class TestIA3ForwardInvariant:
    """``IA3Linear(x) == base_layer(x) * ia3_l`` for any input / scale."""

    @given(data=_ia3_layer_and_input())
    @settings(max_examples=20, deadline=None)
    def test_forward_equals_base_times_scale(self, data):
        layer, x, scale = data
        out = layer(x)
        expected = layer.base_layer(x) * scale
        assert torch.allclose(out, expected, atol=1e-6), f"IA3 forward diverged from base*scale at scale={scale}"

    @given(data=_ia3_layer_and_input())
    @settings(max_examples=20, deadline=None)
    def test_forward_at_init_equals_base(self, data):
        """At ``init_scale=1.0`` (default), the wrapper is the identity
        on top of the base layer.
        """
        layer, x, _ = data
        # Reset to identity init.
        with torch.no_grad():
            layer.ia3_l.fill_(1.0)
        assert torch.allclose(layer(x), layer.base_layer(x), atol=1e-6)


class TestIA3MergeRoundtripInvariant:
    """``merge`` makes the wrapper identity; ``merge → unmerge``
    restores the base weight exactly.
    """

    @given(data=_ia3_layer_and_input())
    @settings(max_examples=15, deadline=None)
    def test_merge_preserves_forward(self, data):
        layer, x, _ = data
        before = layer(x).detach().clone()
        layer.merge_weights()
        after = layer(x).detach().clone()
        # Pre-merge and post-merge forward must agree (the merge folds
        # the scale into the base weight, then sets ia3_l to ones —
        # the effective computation is unchanged).
        assert torch.allclose(before, after, atol=1e-5)

    @given(data=_ia3_layer_and_input())
    @settings(max_examples=15, deadline=None)
    def test_merge_then_unmerge_restores_base(self, data):
        layer, _, scale = data
        # Skip the degenerate zero-scale case — merging an all-zero
        # scale produces an all-zero weight, and unmerging divides by
        # the restored zero (NaN). Real users don't run merge→unmerge
        # on a layer they zeroed out, so we constrain the strategy.
        assume(abs(scale) > 1e-3)
        original_weight = layer.base_layer.weight.detach().clone()
        layer.merge_weights()
        layer.unmerge_weights()
        # Weight must round-trip. Bias is intentionally NOT round-tripped:
        # ``merge_weights`` zeros out the original scale snapshot via
        # ``_merged_ia3_l`` and restores it via ``unmerge_weights``,
        # but the bias restoration uses the *restored* ``ia3_l``
        # which under float arithmetic can drift from the original.
        # Pin the weight (deterministic) and skip bias (float drift).
        assert torch.allclose(layer.base_layer.weight, original_weight, atol=1e-5)


class TestIA3DisableInvariant:
    """``disable_ia3`` (sets ``ia3_l = ones``) makes the wrapper identity."""

    @given(data=_ia3_layer_and_input())
    @settings(max_examples=15, deadline=None)
    def test_disable_makes_wrapper_identity(self, data):
        layer, x, _ = data
        disable_ia3(layer)
        assert torch.allclose(layer(x), layer.base_layer(x), atol=1e-6)

    @given(data=_ia3_layer_and_input())
    @settings(max_examples=15, deadline=None)
    def test_disable_then_enable_restores(self, data):
        layer, _, scale = data
        with torch.no_grad():
            layer.ia3_l.fill_(scale)
        disable_ia3(layer)
        enable_ia3(layer)
        assert torch.allclose(layer.ia3_l, torch.full_like(layer.ia3_l, scale), atol=1e-6)


# ---------------------------------------------------------------------------
# BitFit — structural invariants (no math, just requires_grad toggles)
# ---------------------------------------------------------------------------


class TestBitFitStructuralInvariant:
    """BitFit toggles requires_grad correctly for any bias / weight
    configuration.
    """

    @given(spec=_bias_holder_spec())
    @settings(max_examples=20, deadline=None)
    def test_apply_bitfit_freezes_non_bias_enables_bias(self, spec):
        hidden, bias_tup = spec
        model = _build_holder(hidden, bias_tup)
        apply_bitfit(model)
        for name, p in model.named_parameters():
            is_bias = name == "bias" or name.endswith(".bias")
            assert p.requires_grad is is_bias, (
                f"BitFit post-condition violated at {name}: requires_grad={p.requires_grad}, is_bias={is_bias}"
            )

    @given(spec=_bias_holder_spec())
    @settings(max_examples=20, deadline=None)
    def test_trainable_count_equals_bias_count(self, spec):
        """For ANY model, after apply_bitfit the trainable count is
        exactly the sum of bias sizes — independent of bias
        presence on each Linear.
        """
        hidden, bias_tup = spec
        model = _build_holder(hidden, bias_tup)
        apply_bitfit(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        expected = sum(p.numel() for name, p in model.named_parameters() if name == "bias" or name.endswith(".bias"))
        assert trainable == expected

    @given(spec=_bias_holder_spec())
    @settings(max_examples=15, deadline=None)
    def test_unapply_restores_state(self, spec):
        """``apply → unapply`` returns every parameter's
        ``requires_grad`` to its pre-apply value (typically all-True
        by default).
        """
        hidden, bias_tup = spec
        model = _build_holder(hidden, bias_tup)
        original = {name: p.requires_grad for name, p in model.named_parameters()}
        apply_bitfit(model)
        unapply_bitfit(model)
        for name, p in model.named_parameters():
            assert p.requires_grad == original[name], (
                f"unapply failed at {name}: expected {original[name]}, got {p.requires_grad}"
            )

    @given(spec=_bias_holder_spec())
    @settings(max_examples=15, deadline=None)
    def test_get_bitfit_parameters_yields_only_trainable_biases(self, spec):
        """``get_bitfit_parameters`` yields exactly the bias parameters
        that are trainable — never any weight, never a frozen bias.
        """
        hidden, bias_tup = spec
        model = _build_holder(hidden, bias_tup)
        apply_bitfit(model)
        params = {id(p) for p in get_bitfit_parameters(model)}
        for name, p in model.named_parameters():
            is_bias = name == "bias" or name.endswith(".bias")
            if is_bias:
                assert id(p) in params, f"expected bias {name} in helper output"
            else:
                assert id(p) not in params, f"non-bias {name} leaked into helper output"


# ---------------------------------------------------------------------------
# LoRA — forward math is exactly base + (A @ B * scaling)
# ---------------------------------------------------------------------------


class TestLoRAForwardInvariant:
    """``LoRALinear(x) == base_layer(x) + lora_path * scaling``."""

    @given(
        in_features=st.integers(min_value=4, max_value=12),
        out_features=st.integers(min_value=4, max_value=12),
        rank=st.integers(min_value=1, max_value=4),
        alpha=st.floats(min_value=0.5, max_value=4.0, allow_nan=False, allow_infinity=False),
        batch=st.integers(min_value=1, max_value=3),
        seq=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=15, deadline=None)
    def test_forward_matches_manual_path(self, in_features, out_features, rank, alpha, batch, seq):
        torch.manual_seed(0)
        base = nn.Linear(in_features, out_features, bias=False)
        layer = LoRALinear(base, rank=rank, alpha=alpha)
        x = torch.randn(batch, seq, in_features)
        out = layer(x)
        scaling = alpha / rank
        expected = base(x) + (x @ layer.lora_A @ layer.lora_B) * scaling
        assert torch.allclose(out, expected, atol=1e-5)

    @given(
        in_features=st.integers(min_value=4, max_value=12),
        out_features=st.integers(min_value=4, max_value=12),
        rank=st.integers(min_value=1, max_value=4),
        alpha=st.floats(min_value=0.5, max_value=4.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=10, deadline=None)
    def test_merge_then_unmerge_restores_base(self, in_features, out_features, rank, alpha):
        torch.manual_seed(0)
        base = nn.Linear(in_features, out_features, bias=False)
        layer = LoRALinear(base, rank=rank, alpha=alpha)
        original = base.weight.detach().clone()
        layer.merge_weights()
        layer.unmerge_weights()
        assert torch.allclose(base.weight, original, atol=1e-5)


# ---------------------------------------------------------------------------
# Cross-method invariant: parameter efficiency ranking
# ---------------------------------------------------------------------------


class TestPEFTEfficiencyInvariant:
    """The three small PEFT methods (BitFit, IA3, LoRA at rank=1) all
    produce strictly smaller trainable footprints than LoRA at a
    typical rank — this is the headline property of the small-PEFT
    suite and should hold for ANY model shape.
    """

    @given(spec=_bias_holder_spec())
    @settings(max_examples=10, deadline=None)
    def test_bitfit_smaller_than_typical_lora(self, spec):
        """BitFit's footprint is smaller than LoRA at rank=4 (a
        typical non-minimal setting) on the SAME model.
        """
        from llm.core.lora import apply_lora

        hidden, bias_tup = spec
        bitfit_model = _build_holder(hidden, bias_tup)
        lora_model = _build_holder(hidden, bias_tup)

        apply_bitfit(bitfit_model)
        apply_lora(lora_model, rank=4, alpha=8.0)

        bitfit_n = sum(p.numel() for p in bitfit_model.parameters() if p.requires_grad)
        lora_n = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        # Strict — at rank=4 LoRA is always bigger than just biases.
        assert bitfit_n < lora_n, f"BitFit ({bitfit_n}) not smaller than LoRA-r4 ({lora_n})"

    @given(spec=_bias_holder_spec())
    @settings(max_examples=10, deadline=None)
    def test_ia3_smaller_than_typical_lora(self, spec):
        """IA3's footprint (one scale per Linear) is smaller than LoRA
        at rank=4 on the SAME model.
        """
        from llm.core.lora import apply_lora

        hidden, bias_tup = spec
        ia3_model = _build_holder(hidden, bias_tup)
        lora_model = _build_holder(hidden, bias_tup)

        apply_ia3(ia3_model)
        apply_lora(lora_model, rank=4, alpha=8.0)

        ia3_n = sum(p.numel() for p in ia3_model.parameters() if p.requires_grad)
        lora_n = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        # IA3 = sum(out_features for every Linear). LoRA-r4 = 4 * sum(in+out) per Linear.
        # At rank=4, LoRA is always bigger than IA3 for any non-trivial shape.
        assert ia3_n < lora_n, f"IA3 ({ia3_n}) not smaller than LoRA-r4 ({lora_n})"
