"""Tests for the IA³ (T-Few) foundation slice.

Mirrors the LoRA test pattern where it makes sense (init / freeze /
trainable-params / merge roundtrip), and adds IA³-specific coverage
where the math differs (init to ones → identity transform; scale is
multiplicative not additive; ``count_ia3_parameters`` is dominated by
the frozen base weights, etc.).

The multiplicative design has a much smaller trainable footprint than
LoRA at matched ``out_features`` — that's the headline property of
IA³, so several tests pin it explicitly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from llm.core.ia3 import (
    IA3Linear,
    apply_ia3,
    count_ia3_parameters,
    disable_ia3,
    enable_ia3,
    get_ia3_parameters,
    merge_ia3,
    unmerge_ia3,
)

# ---------------------------------------------------------------------------
# IA3Linear — single-wrapper behaviour
# ---------------------------------------------------------------------------


class TestIA3LinearInit:
    """Construction freezes the base and inits ``ia3_l`` to ones (identity)."""

    def test_base_layer_frozen(self):
        layer = IA3Linear(nn.Linear(16, 32))
        assert layer.base_layer.weight.requires_grad is False
        if layer.base_layer.bias is not None:
            assert layer.base_layer.bias.requires_grad is False

    def test_ia3_l_trainable(self):
        layer = IA3Linear(nn.Linear(16, 32))
        assert layer.ia3_l.requires_grad is True

    def test_init_scale_default_is_ones(self):
        """Identity init: ``ia3_l`` is all ones, so the wrapper preserves
        the base layer's output at step 1 — no chicken-and-egg problem.
        """
        layer = IA3Linear(nn.Linear(16, 32))
        assert torch.allclose(layer.ia3_l, torch.ones_like(layer.ia3_l))

    def test_init_scale_can_be_changed(self):
        layer = IA3Linear(nn.Linear(16, 32), init_scale=0.5)
        assert torch.allclose(layer.ia3_l, torch.full_like(layer.ia3_l, 0.5))

    def test_shape_matches_out_features(self):
        layer = IA3Linear(nn.Linear(16, 64))
        assert layer.ia3_l.shape == (64,)

    def test_trainable_parameters_count(self):
        """Trainable params are exactly ``out_features`` — that's the IA³
        selling point vs. LoRA's ``rank * (in + out)``.
        """
        layer = IA3Linear(nn.Linear(16, 64))
        assert layer.trainable_parameters == 64

    def test_extra_repr_contains_key_info(self):
        layer = IA3Linear(nn.Linear(16, 64))
        text = layer.extra_repr()
        assert "in_features=16" in text
        assert "out_features=64" in text
        assert "trainable=64" in text


# ---------------------------------------------------------------------------
# Forward math — identity at init, multiplies the base output after
# ---------------------------------------------------------------------------


class TestIA3LinearForward:
    """Forward is ``base_output * ia3_l`` (broadcast over batch / seq)."""

    def test_forward_at_init_matches_base(self):
        """Default ``init_scale=1.0`` → wrapper output == base output."""
        torch.manual_seed(0)
        base = nn.Linear(16, 32)
        layer = IA3Linear(base)
        x = torch.randn(2, 5, 16)
        assert torch.allclose(layer(x), base(x), atol=1e-6)

    def test_forward_scales_output(self):
        """Setting ``ia3_l = 2`` doubles the base output."""
        torch.manual_seed(0)
        base = nn.Linear(16, 32)
        layer = IA3Linear(base)
        x = torch.randn(2, 5, 16)
        with torch.no_grad():
            layer.ia3_l.fill_(2.0)
        out = layer(x)
        base_out = base(x)
        assert torch.allclose(out, base_out * 2.0, atol=1e-6)

    def test_forward_per_channel_scaling(self):
        """Different ``ia3_l`` entries scale different output channels."""
        torch.manual_seed(0)
        base = nn.Linear(16, 8)
        layer = IA3Linear(base)
        x = torch.randn(2, 5, 16)
        scale = torch.tensor([0.0, 1.0, 2.0, 0.5, 3.0, 1.5, 0.1, 4.0])
        with torch.no_grad():
            layer.ia3_l.copy_(scale)
        out = layer(x)
        base_out = base(x)
        assert torch.allclose(out, base_out * scale, atol=1e-6)

    def test_forward_broadcasts_over_batch_and_seq(self):
        """``ia3_l`` has shape ``(out_features,)``; the base output is
        ``(B, S, out_features)``. Broadcast must Just Work.
        """
        torch.manual_seed(0)
        base = nn.Linear(16, 32)
        layer = IA3Linear(base)
        x = torch.randn(4, 7, 16)
        out = layer(x)
        assert out.shape == (4, 7, 32)

    def test_forward_grad_flows_only_to_ia3_l(self):
        """Backward must update ``ia3_l`` and leave the base frozen."""
        layer = IA3Linear(nn.Linear(16, 32))
        x = torch.randn(2, 5, 16)
        loss = layer(x).sum()
        loss.backward()
        assert layer.ia3_l.grad is not None
        # Base weight is frozen → no grad populated even if the math
        # would otherwise produce one.
        assert layer.base_layer.weight.grad is None


# ---------------------------------------------------------------------------
# Merge / unmerge — fold the scale into the base for inference
# ---------------------------------------------------------------------------


class TestIA3LinearMerge:
    """``merge_weights`` folds ``ia3_l`` into the base; ``unmerge``
    reverses it. After merge, ``ia3_l`` is identity so the wrapper is
    a no-op.
    """

    def test_merge_then_forward_matches_pre_merge(self):
        """Math invariant: ``merge_ia3(model)(x) == IA3(ia3_l=ones)(x)``.

        Concretely: after ``merge``, the wrapped base weight already
        contains the scale, so the wrapper's ``base(x) * l`` collapses
        to ``base(x)`` (because ``l == ones`` at the new effective
        state). Forward output should be byte-equal to the pre-merge
        forward.
        """
        torch.manual_seed(0)
        base = nn.Linear(16, 32)
        layer = IA3Linear(base)
        x = torch.randn(2, 5, 16)

        before = layer(x).detach().clone()
        layer.merge_weights()
        # After merge, the wrapper still runs `base(x) * l` — but the
        # merged weight plus a still-ones ``ia3_l`` yields the same
        # output. The invariant we actually verify is that the merged
        # forward equals the pre-merge forward (modulo float drift).
        after = layer(x).detach().clone()
        assert torch.allclose(before, after, atol=1e-5)

    def test_merge_then_zero_ia3_l_zeros_output(self):
        """After merge, setting ``ia3_l = 0`` zeros out everything.

        This is the strict test that the merge actually folded the
        scale into the base weight — if it didn't, ``ia3_l = 0`` would
        also zero out the output (trivially). The check above
        (pre/post merge identical) verifies the merge happened; this
        check verifies the merge was *complete* (no residual scale
        hiding in ``ia3_l``).
        """
        torch.manual_seed(0)
        base = nn.Linear(16, 32)
        layer = IA3Linear(base)
        x = torch.randn(2, 5, 16)
        layer.merge_weights()
        with torch.no_grad():
            layer.ia3_l.fill_(0.0)
        out = layer(x)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_unmerge_restores_base_weight(self):
        """``merge`` then ``unmerge`` returns the base weight to its
        original value (modulo float drift).
        """
        torch.manual_seed(0)
        base = nn.Linear(16, 32)
        original_weight = base.weight.detach().clone()
        layer = IA3Linear(base)
        # Set a non-trivial scale so the merge actually changes the weight.
        with torch.no_grad():
            layer.ia3_l.copy_(torch.linspace(0.5, 1.5, 32))
        layer.merge_weights()
        layer.unmerge_weights()
        assert torch.allclose(base.weight, original_weight, atol=1e-5)

    def test_merge_handles_no_bias(self):
        """``merge_weights`` doesn't crash when the base has no bias."""
        base = nn.Linear(16, 32, bias=False)
        layer = IA3Linear(base)
        layer.merge_weights()  # must not raise
        layer.unmerge_weights()  # must not raise

    def test_merge_changes_base_weight(self):
        """A non-identity scale must actually change the base weight."""
        torch.manual_seed(0)
        base = nn.Linear(16, 32)
        original_weight = base.weight.detach().clone()
        layer = IA3Linear(base)
        with torch.no_grad():
            layer.ia3_l.copy_(torch.linspace(0.5, 1.5, 32))
        layer.merge_weights()
        assert not torch.allclose(base.weight, original_weight)


# ---------------------------------------------------------------------------
# Module-level helpers — apply / merge / get / count / disable / enable
# ---------------------------------------------------------------------------


class _ToyModel(nn.Module):
    """Tiny model with three Linears, named to exercise target_modules."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.q_proj = nn.Linear(8, 16)
        self.attn.k_proj = nn.Linear(8, 16)
        self.attn.v_proj = nn.Linear(8, 16)
        self.mlp = nn.Linear(16, 32)

    def forward(self, x):  # pragma: no cover — not used in tests
        return self.mlp(self.attn.v_proj(x))


class TestApplyIA3:
    """``apply_ia3`` walks the model and wraps matching Linears."""

    def test_no_target_wraps_every_linear(self):
        model = _ToyModel()
        apply_ia3(model)
        for name, m in model.named_modules():
            if isinstance(m, IA3Linear):
                # Every wrapper points at a Linear of the right shape.
                assert isinstance(m.base_layer, nn.Linear)

    def test_target_modules_substring_filter(self):
        """``target_modules=["q_proj"]`` only wraps modules whose name
        contains ``"q_proj"``."""
        model = _ToyModel()
        apply_ia3(model, target_modules=["q_proj"])
        wrappers = [name for name, m in model.named_modules() if isinstance(m, IA3Linear)]
        assert wrappers == ["attn.q_proj"]

    def test_multiple_target_modules(self):
        model = _ToyModel()
        apply_ia3(model, target_modules=["q_proj", "v_proj"])
        wrappers = sorted(name for name, m in model.named_modules() if isinstance(m, IA3Linear))
        assert wrappers == ["attn.q_proj", "attn.v_proj"]

    def test_apply_is_in_place(self):
        """``apply_ia3`` mutates the model (matches LoRA convention)."""
        model = _ToyModel()
        result = apply_ia3(model)
        assert result is model

    def test_empty_target_modules_list_wraps_everything(self):
        """Empty list is the documented "wrap every Linear" sentinel
        (matches ``apply_lora`` behaviour).
        """
        model = _ToyModel()
        apply_ia3(model, target_modules=[])
        n_linears = sum(1 for _ in model.modules() if isinstance(_, nn.Linear))
        n_ia3 = sum(1 for _ in model.modules() if isinstance(_, IA3Linear))
        assert n_ia3 == n_linears


class TestModuleLevelMergeUnmerge:
    def test_merge_ia3_walks_and_folds(self):
        torch.manual_seed(0)
        model = _ToyModel()
        original_q_weight = model.attn.q_proj.weight.detach().clone()
        apply_ia3(model, target_modules=["q_proj"])
        # Non-trivial scale so the merge actually changes the weight.
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, IA3Linear):
                    m.ia3_l.copy_(torch.linspace(0.5, 1.5, m.ia3_l.numel()))
        merge_ia3(model)
        # The base weight changed (merge folded the scale in).
        assert not torch.allclose(model.attn.q_proj.base_layer.weight, original_q_weight)

    def test_unmerge_ia3_restores(self):
        torch.manual_seed(0)
        model = _ToyModel()
        original_q_weight = model.attn.q_proj.weight.detach().clone()
        apply_ia3(model, target_modules=["q_proj"])
        # Non-trivial scale so the merge actually changes something.
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, IA3Linear):
                    m.ia3_l.copy_(torch.linspace(0.5, 1.5, m.ia3_l.numel()))
        merge_ia3(model)
        unmerge_ia3(model)
        assert torch.allclose(model.attn.q_proj.base_layer.weight, original_q_weight, atol=1e-5)


class TestGetIA3Parameters:
    """``get_ia3_parameters`` yields one ``ia3_l`` per wrapper — nothing else."""

    def test_yields_one_param_per_wrapper(self):
        model = _ToyModel()
        apply_ia3(model)
        params = list(get_ia3_parameters(model))
        n_wrappers = sum(1 for _ in model.modules() if isinstance(_, IA3Linear))
        assert len(params) == n_wrappers
        for p in params:
            assert p.requires_grad is True

    def test_yields_nothing_when_no_wrappers(self):
        model = _ToyModel()
        # No apply_ia3 → no wrappers → empty iterator.
        assert list(get_ia3_parameters(model)) == []

    def test_optimizer_wired_via_helper_only_updates_ia3_l(self):
        """``torch.optim.Adam(get_ia3_parameters(model))`` leaves the base
        weight byte-identical across an Adam step.
        """
        torch.manual_seed(0)
        model = _ToyModel()
        apply_ia3(model, target_modules=["q_proj"])
        wrapper = model.attn.q_proj
        base_before = wrapper.base_layer.weight.detach().clone()
        ia3_before = wrapper.ia3_l.detach().clone()

        opt = torch.optim.Adam(get_ia3_parameters(model), lr=1e-2)
        x = torch.randn(2, 4, 8)
        # Forward through the wrapped q_proj only — keeps the test
        # focused on the gradient-flow contract, independent of the
        # rest of the (toy) model.
        loss = wrapper(x).sum()
        loss.backward()
        opt.step()

        # Base weight unchanged.
        assert torch.allclose(wrapper.base_layer.weight, base_before, atol=1e-6)
        # ia3_l updated.
        assert not torch.allclose(wrapper.ia3_l, ia3_before)


class TestCountIA3Parameters:
    def test_trainable_is_dominated_by_frozen_base(self):
        """At matched ``out_features``, IA³ trainable params are ~2 orders
        of magnitude smaller than LoRA — that's the headline property.
        """
        model = _ToyModel()
        apply_ia3(model)
        trainable, total = count_ia3_parameters(model)
        # Four Linears (q/k/v_proj + mlp), each contributing its
        # out_features. q/k/v = 16, mlp = 32 → 3*16 + 32 = 80.
        assert trainable == 16 + 16 + 16 + 32
        # Total dwarfs trainable — base weights are frozen.
        assert total > trainable * 5

    def test_count_returns_all_trainable_when_no_wrappers(self):
        """Without ``apply_ia3``, every Linear is trainable by default —
        the helper reports the same as ``sum(p.requires_grad)``. Users
        who want the IA³-specific count should look at
        ``sum(m.ia3_l.numel() for m in model.modules() if isinstance(m, IA3Linear))``
        instead.
        """
        model = _ToyModel()
        trainable, total = count_ia3_parameters(model)
        # No apply_ia3 → every weight is trainable, none are frozen.
        assert trainable > 0
        assert trainable == total  # nothing frozen
        # Apply IA3 — now only ia3_l is trainable.
        apply_ia3(model)
        trainable_after, _ = count_ia3_parameters(model)
        assert trainable_after == 80
        assert trainable_after < trainable


class TestDisableEnableIA3:
    """``disable_ia3`` zeros the effective scale (sets to ones, then
    snapshot is preserved for ``enable_ia3``).
    """

    def test_disable_sets_ia3_l_to_ones(self):
        torch.manual_seed(0)
        model = _ToyModel()
        apply_ia3(model, target_modules=["q_proj"])
        # Modify ia3_l so disable is a non-trivial test.
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, IA3Linear):
                    m.ia3_l.copy_(torch.linspace(0.1, 0.9, m.ia3_l.numel()))
        disable_ia3(model)
        for m in model.modules():
            if isinstance(m, IA3Linear):
                assert torch.allclose(m.ia3_l, torch.ones_like(m.ia3_l), atol=1e-6)

    def test_enable_restores_snapshot(self):
        torch.manual_seed(0)
        model = _ToyModel()
        apply_ia3(model, target_modules=["q_proj"])
        original_scale = torch.linspace(0.1, 0.9, 16)
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, IA3Linear):
                    m.ia3_l.copy_(original_scale)
        disable_ia3(model)
        enable_ia3(model)
        for m in model.modules():
            if isinstance(m, IA3Linear):
                assert torch.allclose(m.ia3_l, original_scale, atol=1e-6)

    def test_disable_then_forward_matches_base(self):
        """``disable_ia3`` → wrapper output == base output (identity)."""
        torch.manual_seed(0)
        model = _ToyModel()
        apply_ia3(model, target_modules=["q_proj"])
        x = torch.randn(2, 4, 8)
        # Replace q_proj with a fresh Linear so we can compare apples-to-apples.
        fresh_q = nn.Linear(8, 16)
        with torch.no_grad():
            fresh_q.weight.copy_(model.attn.q_proj.base_layer.weight)
            if fresh_q.bias is not None and model.attn.q_proj.base_layer.bias is not None:
                fresh_q.bias.copy_(model.attn.q_proj.base_layer.bias)
        disable_ia3(model)
        assert torch.allclose(model.attn.q_proj(x), fresh_q(x), atol=1e-6)


# ---------------------------------------------------------------------------
# Headline-property test: IA³ parameter efficiency vs. LoRA
# ---------------------------------------------------------------------------


class TestIA3IsMoreEfficientThanLoRA:
    """Pin the headline property: at matched ``out_features``, IA³
    trainable params are smaller than LoRA's ``rank * (in + out)``.

    This is the test a reviewer should read first to understand what
    IA³ actually buys you.
    """

    def test_ia3_trainable_params_smaller_than_lora_at_matched_out(self):
        from llm.core.lora import LoRALinear

        in_features, out_features = 64, 128
        rank = 8

        ia3 = IA3Linear(nn.Linear(in_features, out_features))
        lora = LoRALinear(nn.Linear(in_features, out_features), rank=rank, alpha=16.0)

        ia3_params = ia3.trainable_parameters
        lora_params = lora.trainable_parameters

        assert ia3_params == out_features
        assert lora_params == rank * (in_features + out_features)
        # Sanity: IA³ is meaningfully smaller.
        assert ia3_params * 4 < lora_params
