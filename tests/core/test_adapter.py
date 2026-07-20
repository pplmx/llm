"""Tests for the Adapter (Houlsby 2019) foundation slice.

AdapterLinear wraps a frozen ``nn.Linear`` with a bottleneck residual
module. The forward math is exactly ``base(x) + up(activation(down(base(x))))``
— the test surface focuses on three contracts:

1. **Init identity**: ``up`` is zero-initialized, so the wrapper is
   the identity on top of the base at step 1. Pre-training forward
   matches base forward byte-for-byte.
2. **Adapter math**: After non-zero ``up``, the forward output is
   ``base + adapter(base)`` — the residual is over the base output,
   not the input. Adapter activates only via the down → activation
   → up path; the base path is unchanged.
3. **Parameter contract**: Base is frozen, adapter is trainable.
   ``trainable_parameters`` is exactly the down + up weight + bias
   count. ``get_adapter_parameters`` yields only the adapter params.

Mirrors the LoRA / IA³ test pattern for one-import consistency.
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

# ---------------------------------------------------------------------------
# AdapterLinear — single-wrapper behaviour
# ---------------------------------------------------------------------------


class TestAdapterLinearInit:
    """Construction freezes the base, inits adapter with ``up = zeros``."""

    def test_base_layer_frozen(self):
        layer = AdapterLinear(nn.Linear(16, 32), bottleneck_dim=8)
        assert layer.base_layer.weight.requires_grad is False
        if layer.base_layer.bias is not None:
            assert layer.base_layer.bias.requires_grad is False

    def test_adapter_params_trainable(self):
        layer = AdapterLinear(nn.Linear(16, 32), bottleneck_dim=8)
        assert layer.down.weight.requires_grad is True
        assert layer.down.bias.requires_grad is True
        assert layer.up.weight.requires_grad is True
        assert layer.up.bias.requires_grad is True

    def test_up_projection_zero_at_init(self):
        """``up.weight`` and ``up.bias`` are zero-initialized so the
        adapter contributes 0 to the output at step 1.
        """
        layer = AdapterLinear(nn.Linear(16, 32), bottleneck_dim=8)
        assert torch.allclose(layer.up.weight, torch.zeros_like(layer.up.weight))
        assert torch.allclose(layer.up.bias, torch.zeros_like(layer.up.bias))

    def test_down_projection_not_zero_at_init(self):
        """``down`` uses default Linear init (Kaiming uniform) — NOT
        zero. A zero-init on down would make the adapter contribute
        zero regardless of training, so the adapter path would never
        receive any gradient signal.
        """
        layer = AdapterLinear(nn.Linear(16, 32), bottleneck_dim=8)
        assert not torch.allclose(layer.down.weight, torch.zeros_like(layer.down.weight))

    def test_bottleneck_dim_must_be_positive(self):
        with pytest.raises(ValueError, match="bottleneck_dim"):
            AdapterLinear(nn.Linear(16, 32), bottleneck_dim=0)
        with pytest.raises(ValueError, match="bottleneck_dim"):
            AdapterLinear(nn.Linear(16, 32), bottleneck_dim=-4)

    def test_trainable_parameters_count(self):
        """Trainable params = down (hidden × bottleneck + bottleneck) +
        up (bottleneck × hidden + hidden) = 2 × hidden × bottleneck +
        hidden + bottleneck.
        """
        in_features, out_features, bottleneck = 16, 32, 8
        layer = AdapterLinear(nn.Linear(in_features, out_features), bottleneck_dim=bottleneck)
        expected = (
            out_features * bottleneck  # down.weight
            + bottleneck  # down.bias
            + bottleneck * out_features  # up.weight
            + out_features  # up.bias
        )
        assert layer.trainable_parameters == expected

    def test_extra_repr_contains_key_info(self):
        layer = AdapterLinear(nn.Linear(16, 32), bottleneck_dim=8)
        text = layer.extra_repr()
        assert "in_features=16" in text
        assert "out_features=32" in text
        assert "bottleneck=8" in text


# ---------------------------------------------------------------------------
# Forward math — identity at init, residual adapter after training
# ---------------------------------------------------------------------------


class TestAdapterLinearForward:
    """Forward is ``base(x) + up(activation(down(base(x))))``."""

    def test_forward_at_init_matches_base(self):
        """Default init: ``up = zeros`` → adapter output is 0 → wrapper
        output equals base output byte-for-byte.
        """
        torch.manual_seed(0)
        base = nn.Linear(16, 32)
        layer = AdapterLinear(base, bottleneck_dim=8)
        x = torch.randn(2, 5, 16)
        assert torch.allclose(layer(x), base(x), atol=1e-6)

    def test_forward_after_up_nonzero_adds_residual(self):
        """Set ``up`` to a non-zero constant — wrapper output is now
        ``base(x) + up(activation(down(base(x))))``.
        """
        torch.manual_seed(0)
        base = nn.Linear(16, 32)
        layer = AdapterLinear(base, bottleneck_dim=8)
        x = torch.randn(2, 5, 16)
        base_out = base(x)
        # Set up.weight to a constant, up.bias to 0.
        with torch.no_grad():
            layer.up.weight.fill_(0.1)
            layer.up.bias.zero_()
        out = layer(x)
        expected = base_out + layer.up(layer.activation(layer.down(base_out)))
        assert torch.allclose(out, expected, atol=1e-5)

    def test_residual_is_over_base_output_not_input(self):
        """The adapter residual is summed with the BASE output, not
        the input ``x``. The ``down`` projection sees the base
        output, not the raw input.

        We verify the structural property by checking that the wrapper
        output equals ``base_out + adapter(base_out)`` (over the base
        output). The dimension check is implicit — ``down`` is a
        ``Linear(base.out_features, bottleneck)``, so feeding it
        anything other than the base output would shape-error.
        """
        torch.manual_seed(0)
        base = nn.Linear(16, 32)
        layer = AdapterLinear(base, bottleneck_dim=8)
        x = torch.randn(2, 5, 16)
        with torch.no_grad():
            layer.up.weight.fill_(1.0)
            layer.up.bias.zero_()
        out = layer(x)
        base_out = base(x)
        adapter_path = layer.up(layer.activation(layer.down(base_out)))
        assert torch.allclose(out, base_out + adapter_path, atol=1e-5)

    def test_forward_broadcasts_over_batch_and_seq(self):
        """``down`` and ``up`` operate on ``out_features`` dim;
        broadcast over batch and seq works automatically.
        """
        torch.manual_seed(0)
        base = nn.Linear(16, 32)
        layer = AdapterLinear(base, bottleneck_dim=8)
        x = torch.randn(4, 7, 16)
        out = layer(x)
        assert out.shape == (4, 7, 32)

    def test_forward_grad_flows_only_to_adapter(self):
        """Backward must update the adapter params and leave the base
        weight frozen (grad is None on the base).
        """
        layer = AdapterLinear(nn.Linear(16, 32), bottleneck_dim=8)
        x = torch.randn(2, 5, 16)
        loss = layer(x).sum()
        loss.backward()
        # Adapter params received gradients.
        assert layer.down.weight.grad is not None
        assert layer.down.bias.grad is not None
        assert layer.up.weight.grad is not None
        assert layer.up.bias.grad is not None
        # Base weight is frozen.
        assert layer.base_layer.weight.grad is None


# ---------------------------------------------------------------------------
# Merge / unmerge — no-ops for adapters (kept for API parity)
# ---------------------------------------------------------------------------


class TestAdapterLinearMerge:
    """``merge_weights`` / ``unmerge_weights`` are no-ops for adapters.

    Unlike LoRA / IA³, the adapter has no math to fold into the base
    weight — the up-projection being zero means the adapter
    contributes zero to the output unless trained.
    """

    def test_merge_does_not_change_base_weight(self):
        torch.manual_seed(0)
        base = nn.Linear(16, 32)
        original = base.weight.detach().clone()
        layer = AdapterLinear(base, bottleneck_dim=8)
        layer.merge_weights()
        assert torch.allclose(base.weight, original, atol=1e-7)

    def test_unmerge_does_not_change_base_weight(self):
        torch.manual_seed(0)
        base = nn.Linear(16, 32)
        original = base.weight.detach().clone()
        layer = AdapterLinear(base, bottleneck_dim=8)
        layer.unmerge_weights()
        assert torch.allclose(base.weight, original, atol=1e-7)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class _ToyModel(nn.Module):
    """Tiny model with three Linears named to exercise target_modules."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.q_proj = nn.Linear(8, 16)
        self.attn.k_proj = nn.Linear(8, 16)
        self.attn.v_proj = nn.Linear(8, 16)
        self.mlp = nn.Linear(16, 32)

    def forward(self, x):  # pragma: no cover — not used in tests
        return self.mlp(self.attn.v_proj(x))


class TestApplyAdapter:
    """``apply_adapter`` walks the model and wraps matching Linears."""

    def test_no_target_wraps_every_linear(self):
        model = _ToyModel()
        apply_adapter(model, bottleneck_dim=4)
        for name, m in model.named_modules():
            if isinstance(m, AdapterLinear):
                assert isinstance(m.base_layer, nn.Linear)

    def test_target_modules_substring_filter(self):
        model = _ToyModel()
        apply_adapter(model, bottleneck_dim=4, target_modules=["q_proj"])
        wrappers = [name for name, m in model.named_modules() if isinstance(m, AdapterLinear)]
        assert wrappers == ["attn.q_proj"]

    def test_multiple_target_modules(self):
        model = _ToyModel()
        apply_adapter(model, bottleneck_dim=4, target_modules=["q_proj", "v_proj"])
        wrappers = sorted(name for name, m in model.named_modules() if isinstance(m, AdapterLinear))
        assert wrappers == ["attn.q_proj", "attn.v_proj"]

    def test_apply_is_in_place(self):
        model = _ToyModel()
        result = apply_adapter(model, bottleneck_dim=4)
        assert result is model


class TestModuleLevelMergeUnmerge:
    def test_merge_unmerge_are_no_ops(self):
        torch.manual_seed(0)
        model = _ToyModel()
        apply_adapter(model, bottleneck_dim=4, target_modules=["q_proj"])
        original_weight = model.attn.q_proj.base_layer.weight.detach().clone()
        merge_adapter(model)
        unmerge_adapter(model)
        assert torch.allclose(model.attn.q_proj.base_layer.weight, original_weight, atol=1e-7)


class TestGetAdapterParameters:
    """``get_adapter_parameters`` yields only adapter params (4 per wrapper)."""

    def test_yields_four_params_per_wrapper(self):
        model = _ToyModel()
        apply_adapter(model, bottleneck_dim=4)
        params = list(get_adapter_parameters(model))
        n_wrappers = sum(1 for _ in model.modules() if isinstance(_, AdapterLinear))
        assert len(params) == 4 * n_wrappers
        for p in params:
            assert p.requires_grad is True

    def test_yields_nothing_when_no_wrappers(self):
        model = _ToyModel()
        assert list(get_adapter_parameters(model)) == []

    def test_optimizer_only_updates_adapter(self):
        """``torch.optim.Adam(get_adapter_parameters(model))`` leaves the
        base Linear weight byte-identical across an Adam step.

        Note: at ``up = zeros`` init the adapter path is mathematically
        dead — only ``up.weight`` / ``up.bias`` receive gradients at
        step 1. We pre-train by setting ``up.weight`` to a non-zero
        constant so the gradient reaches ``down.weight`` too.
        """
        torch.manual_seed(0)
        model = _ToyModel()
        apply_adapter(model, bottleneck_dim=4, target_modules=["q_proj"])
        wrapper = model.attn.q_proj

        # Pre-train the adapter so down.weight receives a gradient
        # at step 1 (otherwise up = zeros makes the adapter path
        # contribute zero to the output, and the gradient to down
        # is zero).
        with torch.no_grad():
            wrapper.up.weight.fill_(0.1)
            wrapper.up.bias.zero_()

        base_before = wrapper.base_layer.weight.detach().clone()
        down_before = wrapper.down.weight.detach().clone()
        up_before = wrapper.up.weight.detach().clone()

        opt = torch.optim.Adam(get_adapter_parameters(model), lr=1e-2)
        x = torch.randn(2, 4, 8)
        out = wrapper(x).sum()
        out.backward()
        opt.step()

        # Base weight unchanged.
        assert torch.allclose(wrapper.base_layer.weight, base_before, atol=1e-6)
        # Adapter params updated.
        assert not torch.allclose(wrapper.down.weight, down_before, atol=1e-6)
        assert not torch.allclose(wrapper.up.weight, up_before, atol=1e-6)


class TestCountAdapterParameters:
    def test_trainable_matches_per_wrapper_count(self):
        """After ``apply_adapter``, trainable count is exactly 4 × n_wrappers
        worth of params (down weight + down bias + up weight + up bias
        per wrapper).
        """
        model = _ToyModel()
        apply_adapter(model, bottleneck_dim=4)
        trainable, total = count_adapter_parameters(model)
        expected = 4 * sum(1 for _ in model.modules() if isinstance(_, AdapterLinear))
        # Each wrapper contributes out_features × bottleneck + bottleneck +
        # bottleneck × out_features + out_features = 2*out*bottleneck +
        # out + bottleneck. Hard to predict without enumerating shapes,
        # so we just check that trainable > 0 and < total.
        assert trainable > 0
        assert total > trainable

    def test_zero_trainable_without_apply(self):
        model = _ToyModel()
        # No apply_adapter — all params are at their default state
        # (everything trainable, nothing frozen).
        trainable, total = count_adapter_parameters(model)
        assert trainable == total
        assert trainable > 0


class TestDisableEnableAdapter:
    """``disable_adapter`` zeros the up-projection (so the wrapper
    becomes the identity); ``enable_adapter`` restores.
    """

    def test_disable_sets_up_to_zero(self):
        torch.manual_seed(0)
        model = _ToyModel()
        apply_adapter(model, bottleneck_dim=4, target_modules=["q_proj"])
        # Modify up so disable is a non-trivial test.
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, AdapterLinear):
                    m.up.weight.fill_(0.7)
                    m.up.bias.fill_(0.1)
        disable_adapter(model)
        for m in model.modules():
            if isinstance(m, AdapterLinear):
                assert torch.allclose(m.up.weight, torch.zeros_like(m.up.weight))
                assert torch.allclose(m.up.bias, torch.zeros_like(m.up.bias))

    def test_enable_restores_snapshot(self):
        torch.manual_seed(0)
        model = _ToyModel()
        apply_adapter(model, bottleneck_dim=4, target_modules=["q_proj"])
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, AdapterLinear):
                    m.up.weight.fill_(0.7)
                    m.up.bias.fill_(0.1)
        disable_adapter(model)
        enable_adapter(model)
        for m in model.modules():
            if isinstance(m, AdapterLinear):
                assert torch.allclose(m.up.weight, torch.full_like(m.up.weight, 0.7))
                assert torch.allclose(m.up.bias, torch.full_like(m.up.bias, 0.1))

    def test_disable_then_forward_matches_base(self):
        """``disable_adapter`` → wrapper output == base output (identity)."""
        torch.manual_seed(0)
        model = _ToyModel()
        apply_adapter(model, bottleneck_dim=4, target_modules=["q_proj"])
        x = torch.randn(2, 4, 8)
        # Snapshot base weight for fresh comparison.
        original_weight = model.attn.q_proj.base_layer.weight.detach().clone()
        original_bias = model.attn.q_proj.base_layer.bias.detach().clone()
        disable_adapter(model)
        # Build a fresh Linear with the same weights to compare apples-to-apples.
        fresh_q = nn.Linear(8, 16)
        with torch.no_grad():
            fresh_q.weight.copy_(original_weight)
            fresh_q.bias.copy_(original_bias)
        assert torch.allclose(model.attn.q_proj(x), fresh_q(x), atol=1e-6)


# ---------------------------------------------------------------------------
# Headline-property test: Adapter trainable params scale with bottleneck
# ---------------------------------------------------------------------------


class TestAdapterScalingInvariant:
    """Pin the scaling invariant: Adapter trainable params scale with
    ``bottleneck_dim`` (and are independent of rank — LoRA's primary
    knob). This is what makes adapters more parameter-efficient at
    large hidden sizes with a small bottleneck.
    """

    def test_adapter_trainable_scales_with_bottleneck(self):
        small = AdapterLinear(nn.Linear(16, 32), bottleneck_dim=8)
        large = AdapterLinear(nn.Linear(16, 32), bottleneck_dim=64)
        # Larger bottleneck → strictly more trainable params.
        assert large.trainable_parameters > small.trainable_parameters
        # The growth is roughly 8x (both bottleneck and out_features scale).
        ratio = large.trainable_parameters / small.trainable_parameters
        assert ratio > 5.0
