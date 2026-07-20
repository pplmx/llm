"""Tests for the BitFit (bias-only fine-tuning) foundation slice.

BitFit is the simplest PEFT method — no wrappers, no math, just
``requires_grad`` toggling. The test surface focuses on three
contracts:

1. ``apply_bitfit`` freezes everything AND enables every bias (the
   post-condition is what users actually depend on).
2. ``unapply_bitfit`` restores the pre-BitFit ``requires_grad`` state
   exactly — including a model that had a mix of trainable / frozen
   parameters before BitFit was applied.
3. ``get_bitfit_parameters`` yields exactly the bias parameters that
   are trainable, so an optimizer wired via the helper updates only
   biases (base weights stay byte-identical across an Adam step).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from llm.core.bitfit import (
    apply_bitfit,
    count_bitfit_parameters,
    get_bitfit_parameters,
    is_bitfit_applied,
    unapply_bitfit,
)

# ---------------------------------------------------------------------------
# Tiny test fixtures
# ---------------------------------------------------------------------------


class _ToyModel(nn.Module):
    """Tiny model with a bias-bearing Linear and a bias-free Linear.

    Two Linears (``fc_with_bias`` with bias=True, ``fc_no_bias`` with
    bias=False) plus a LayerNorm — that gives us three params with
    "bias" in the name (the Linear's bias and the LayerNorm's bias /
    weight — note: the LayerNorm weight is named ``weight`` not
    ``bias``) and one param without (the Linear's weight).

    Wait — the LayerNorm's ``weight`` is NOT a bias, so we should only
    have TWO "bias"-named params (the Linear's bias and the
    LayerNorm's bias). The LayerNorm weight is a per-feature gain and
    is NOT trainable under BitFit.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc_with_bias = nn.Linear(8, 16, bias=True)
        self.fc_no_bias = nn.Linear(16, 8, bias=False)
        self.norm = nn.LayerNorm(8)

    def forward(self, x):  # pragma: no cover — not used in tests
        return self.norm(self.fc_no_bias(self.fc_with_bias(x)))


class _ModelWithFrozenWeights(nn.Module):
    """Pre-frozen model used to verify unapply_bitfit restores state."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 16, bias=True)
        # Freeze the weight BEFORE BitFit is applied.
        self.fc.weight.requires_grad = False
        # Bias starts trainable.

    def forward(self, x):  # pragma: no cover — not used in tests
        return self.fc(x)


# ---------------------------------------------------------------------------
# apply_bitfit — the main entry point
# ---------------------------------------------------------------------------


class TestApplyBitFit:
    """``apply_bitfit`` freezes everything and enables biases."""

    def test_freezes_all_non_bias_parameters(self):
        model = _ToyModel()
        apply_bitfit(model)
        # Both weights (fc_with_bias.weight, fc_no_bias.weight,
        # norm.weight) are frozen.
        assert model.fc_with_bias.weight.requires_grad is False
        assert model.fc_no_bias.weight.requires_grad is False
        assert model.norm.weight.requires_grad is False

    def test_enables_all_bias_parameters(self):
        model = _ToyModel()
        apply_bitfit(model)
        assert model.fc_with_bias.bias.requires_grad is True
        assert model.norm.bias.requires_grad is True

    def test_no_bias_no_trainable(self):
        """A model with NO bias params at all → ``apply_bitfit`` freezes
        everything (zero trainable params).
        """

        class _BiasFree(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(8, 16, bias=False)

        model = _BiasFree()
        apply_bitfit(model)
        assert model.fc.weight.requires_grad is False
        # No bias to enable.
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == 0

    def test_target_modules_filters_biases(self):
        """``target_modules=["fc_with_bias"]`` only enables biases
        whose qualified name contains the substring.
        """
        model = _ToyModel()
        apply_bitfit(model, target_modules=["fc_with_bias"])
        # fc_with_bias.bias matches → trainable.
        assert model.fc_with_bias.bias.requires_grad is True
        # norm.bias does NOT match → frozen.
        assert model.norm.bias.requires_grad is False

    def test_multiple_target_modules(self):
        model = _ToyModel()
        apply_bitfit(model, target_modules=["fc_with_bias", "norm"])
        assert model.fc_with_bias.bias.requires_grad is True
        assert model.norm.bias.requires_grad is True

    def test_is_in_place(self):
        """``apply_bitfit`` mutates the model (matches LoRA / IA³ convention)."""
        model = _ToyModel()
        result = apply_bitfit(model)
        assert result is model

    def test_idempotent(self):
        """Calling ``apply_bitfit`` twice converges to the same state."""
        model = _ToyModel()
        apply_bitfit(model)
        apply_bitfit(model)
        # Same trainable set, same frozen set. Build a name→param map
        # so we don't have to do an O(n^2) name lookup.
        name_for = {id(p): name for name, p in model.named_parameters()}
        for p in model.parameters():
            name = name_for[id(p)]
            if name == "bias" or name.endswith(".bias"):
                assert p.requires_grad is True
            else:
                assert p.requires_grad is False

    def test_marks_model_as_bitfit_applied(self):
        model = _ToyModel()
        assert is_bitfit_applied(model) is False
        apply_bitfit(model)
        assert is_bitfit_applied(model) is True


# ---------------------------------------------------------------------------
# unapply_bitfit — restore the pre-BitFit state
# ---------------------------------------------------------------------------


class TestUnapplyBitFit:
    """``unapply_bitfit`` reverses the requires_grad changes exactly."""

    def test_restores_all_trainable_when_pre_bitfit_all_trainable(self):
        model = _ToyModel()
        # Pre-BitFit: everything is trainable (default).
        apply_bitfit(model)
        unapply_bitfit(model)
        # All trainable again.
        for p in model.parameters():
            assert p.requires_grad is True

    def test_restores_frozen_weight_when_pre_bitfit_had_frozen(self):
        """Pre-BitFit had ``fc.weight.requires_grad = False``;
        ``apply_bitfit`` doesn't snapshot per-name, but it DOES
        snapshot the exact requires_grad value — so unapply restores
        the original ``False`` even though BitFit also set it to
        ``False``.
        """
        model = _ModelWithFrozenWeights()
        assert model.fc.weight.requires_grad is False  # pre-Bitfit
        assert model.fc.bias.requires_grad is True  # pre-Bitfit
        apply_bitfit(model)
        # Bias frozen by unapply path? No — apply enables it. But the
        # snapshot was saved BEFORE apply changed anything.
        assert model.fc.bias.requires_grad is True
        unapply_bitfit(model)
        # Restored to pre-Bitfit.
        assert model.fc.weight.requires_grad is False
        assert model.fc.bias.requires_grad is True

    def test_unapply_clears_snapshot(self):
        model = _ToyModel()
        apply_bitfit(model)
        assert is_bitfit_applied(model) is True
        unapply_bitfit(model)
        assert is_bitfit_applied(model) is False

    def test_unapply_no_op_when_never_applied(self):
        """``unapply_bitfit`` on a never-applied model is a no-op."""
        model = _ToyModel()
        # No exception, no mutation.
        unapply_bitfit(model)
        # All still trainable.
        for p in model.parameters():
            assert p.requires_grad is True

    def test_reapply_after_unapply_works(self):
        """apply → unapply → apply should converge to the same state as
        a single apply.
        """
        model = _ToyModel()
        apply_bitfit(model)
        unapply_bitfit(model)
        apply_bitfit(model)
        # Same state as a single apply.
        assert model.fc_with_bias.bias.requires_grad is True
        assert model.norm.bias.requires_grad is True
        assert model.fc_with_bias.weight.requires_grad is False
        assert model.fc_no_bias.weight.requires_grad is False
        assert model.norm.weight.requires_grad is False


# ---------------------------------------------------------------------------
# get_bitfit_parameters — what the optimizer sees
# ---------------------------------------------------------------------------


class TestGetBitFitParameters:
    """``get_bitfit_parameters`` yields exactly the bias parameters."""

    def test_yields_only_bias_parameters(self):
        model = _ToyModel()
        apply_bitfit(model)
        params = list(get_bitfit_parameters(model))
        # Two bias params: fc_with_bias.bias, norm.bias.
        assert len(params) == 2
        for p in params:
            assert p.requires_grad is True

    def test_yields_nothing_when_no_biases(self):
        """A bias-free model yields nothing from the helper."""
        model = _ToyModel()
        # Remove the LayerNorm (which has a bias) and use bias=False
        # on the Linear.
        model.fc_with_bias.bias = None
        model.norm = nn.Identity()
        # Recreate apply_bitfit for the modified model.
        apply_bitfit(model)
        params = list(get_bitfit_parameters(model))
        # No biases → no params from the helper.
        assert len(params) == 0

    def test_optimizer_wired_via_helper_only_updates_biases(self):
        """``torch.optim.AdamW(get_bitfit_parameters(model))`` leaves
        the base weight byte-identical across an Adam step.
        """
        torch.manual_seed(0)
        model = _ToyModel()
        apply_bitfit(model)

        weight_before = model.fc_with_bias.weight.detach().clone()
        bias_before = model.fc_with_bias.bias.detach().clone()

        opt = torch.optim.AdamW(get_bitfit_parameters(model), lr=1e-2)

        x = torch.randn(2, 4, 8)
        out = model(x).sum()
        out.backward()
        opt.step()

        # Base weight unchanged (no grad → Adam is a no-op).
        assert torch.allclose(model.fc_with_bias.weight, weight_before, atol=1e-6)
        # Bias updated.
        assert not torch.allclose(model.fc_with_bias.bias, bias_before, atol=1e-6)

    def test_helper_robust_to_manual_non_bias_grad(self):
        """If a user manually enables a non-bias parameter AFTER
        ``apply_bitfit``, ``get_bitfit_parameters`` still yields only
        biases — the suffix check guards against this.
        """
        model = _ToyModel()
        apply_bitfit(model)
        # User manually enables a non-bias parameter.
        model.fc_with_bias.weight.requires_grad = True
        params = list(get_bitfit_parameters(model))
        # Helper should NOT yield the manually-enabled weight, only biases.
        name_for = {id(p): name for name, p in model.named_parameters()}
        for p in params:
            name = name_for[id(p)]
            assert name == "bias" or name.endswith(".bias")


# ---------------------------------------------------------------------------
# count_bitfit_parameters
# ---------------------------------------------------------------------------


class TestCountBitFitParameters:
    """``count_bitfit_parameters`` reports the BitFit contribution."""

    def test_trainable_is_sum_of_biases_after_apply(self):
        model = _ToyModel()
        apply_bitfit(model)
        trainable, total = count_bitfit_parameters(model)
        # Two biases: fc_with_bias.bias (16) + norm.bias (8) = 24.
        assert trainable == 16 + 8
        # Total dwarfs trainable.
        assert total > trainable * 5

    def test_trainable_is_zero_when_no_biases(self):
        """A model with NO biases → 0 trainable after apply."""

        class _BiasFree(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(8, 16, bias=False)

        model = _BiasFree()
        apply_bitfit(model)
        trainable, total = count_bitfit_parameters(model)
        assert trainable == 0
        assert total > 0


# ---------------------------------------------------------------------------
# Headline-property test: BitFit is the smallest possible PEFT method
# ---------------------------------------------------------------------------


class TestBitFitIsSmallestPEFT:
    """Pin the headline property: BitFit has the fewest trainable params
    of any additive PEFT method in the stack — just biases, no matrices,
    no scales.
    """

    def test_bitfit_trainable_strictly_less_than_lora(self):
        """BitFit (sum of bias sizes) is always smaller than LoRA
        (rank * (in + out) per Linear) at typical scales.
        """
        from llm.core.lora import apply_lora

        model = _ToyModel()
        apply_bitfit(model)
        bitfit_trainable, _ = count_bitfit_parameters(model)

        model2 = _ToyModel()
        apply_lora(model2, rank=4, alpha=8.0)
        lora_trainable = sum(p.numel() for p in model2.parameters() if p.requires_grad)

        # LoRA's per-Linear cost is rank * (in + out) — at rank=4 the
        # cost is roughly 4x BitFit's footprint in this toy.
        assert bitfit_trainable < lora_trainable

    def test_bitfit_trainable_matches_bias_count(self):
        """Pin the post-condition: BitFit trainable = sum of bias sizes."""
        model = _ToyModel()
        apply_bitfit(model)
        trainable, _ = count_bitfit_parameters(model)
        # fc_with_bias.bias (16) + norm.bias (8) = 24. fc_no_bias has no bias.
        assert trainable == 24

    def test_bitfit_no_wrappers_no_new_params(self):
        """BitFit adds no new parameters — it just toggles requires_grad.
        This is what makes it the cheapest possible PEFT method.
        """
        model = _ToyModel()
        before_total = sum(p.numel() for p in model.parameters())
        apply_bitfit(model)
        after_total = sum(p.numel() for p in model.parameters())
        # Total param count is unchanged.
        assert before_total == after_total
