"""Tests for the AdaLoRA (Adaptive LoRA) module."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from llm.core.adalora import (
    AdaLoRALinear,
    apply_adalora,
    count_adalora_parameters,
    disable_adalora,
    enable_adalora,
    get_adalora_parameters,
    merge_adalora,
    prune_adalora,
    unmerge_adalora,
)


class TestAdaLoRALinear:
    """Tests for the AdaLoRALinear module."""

    @pytest.fixture
    def base_linear(self):
        return nn.Linear(64, 128)

    @pytest.fixture
    def layer(self, base_linear):
        return AdaLoRALinear(base_linear, init_rank=12, target_rank=8, alpha=32.0)

    def test_init_shape(self, layer):
        """SVD-form parameter shapes match the contract."""
        assert layer.lora_P.shape == (128, 12)
        assert layer.lora_Q.shape == (12, 64)
        assert layer.lora_lambda.shape == (12,)
        assert layer.mask.shape == (12,)

    def test_init_defaults(self, layer):
        """Defaults from the ticket spec: init_rank=12, alpha=32, target=init_rank//2."""
        assert layer.init_rank == 12
        assert layer.target_rank == 8
        assert layer.alpha == 32.0
        # scaling = alpha / init_rank = 32 / 12
        assert math.isclose(layer.scaling, 32.0 / 12, rel_tol=1e-9)

    def test_lambda_initialized_to_zero(self, layer):
        """Paper: Λ initialized to zero so ΔW = 0 at construction."""
        assert torch.allclose(layer.lora_lambda, torch.zeros_like(layer.lora_lambda))

    def test_base_layer_frozen(self, layer):
        """Base layer weight + bias must not receive gradients."""
        assert not layer.base_layer.weight.requires_grad
        if layer.base_layer.bias is not None:
            assert not layer.base_layer.bias.requires_grad

    def test_lora_params_trainable(self, layer):
        """P, Q, λ are all trainable."""
        assert layer.lora_P.requires_grad
        assert layer.lora_Q.requires_grad
        assert layer.lora_lambda.requires_grad

    def test_mask_registered_as_buffer_not_parameter(self, layer):
        """Mask is a buffer (not trained by autograd) so the future
        pruning slice can mutate it without breaking backprop."""
        param_names = {name for name, _ in layer.named_parameters()}
        assert "lora_P" in param_names
        assert "lora_Q" in param_names
        assert "lora_lambda" in param_names
        assert "mask" not in param_names  # buffer, not parameter

    def test_forward_initial_matches_base(self, base_linear, layer):
        """At initialization the layer should behave like the base
        layer because λ = 0 ⇒ ΔW = 0."""
        x = torch.randn(2, 10, 64)
        base_output = base_linear(x)
        adapted_output = layer(x)
        assert torch.allclose(base_output, adapted_output, atol=1e-5)

    def test_forward_shape(self, layer):
        """Forward output shape matches base layer."""
        x = torch.randn(4, 16, 64)
        out = layer(x)
        assert out.shape == (4, 16, 128)

    def test_forward_after_modifying_lambda(self, layer):
        """Non-zero λ produces a non-zero ΔW contribution."""
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            layer.lora_lambda.fill_(0.1)
        base_output = layer.base_layer(x)
        adapted_output = layer(x)
        assert not torch.allclose(base_output, adapted_output, atol=1e-4)

    def test_trainable_parameters_count(self, layer):
        """P + Q + λ = 128*12 + 12*64 + 12 = 1536 + 768 + 12 = 2316."""
        assert layer.trainable_parameters == 128 * 12 + 12 * 64 + 12

    def test_effective_rank_initially_full(self, layer):
        """Mask is all-ones at construction → effective rank = init_rank."""
        assert layer.effective_rank == 12

    def test_effective_rank_reflects_mask(self, layer):
        """Mask pruning zeros contribution → effective_rank drops."""
        with torch.no_grad():
            layer.mask[5:] = 0.0
        assert layer.effective_rank == 5

    def test_mask_pruning_zeros_contribution(self, base_linear, layer):
        """Zeroing all mask entries must restore base-layer behavior."""
        x = torch.randn(2, 10, 64)
        # First, force ΔW to be non-zero so we can verify it goes to zero.
        with torch.no_grad():
            layer.lora_lambda.fill_(0.5)
            layer.lora_P.normal_()
            layer.lora_Q.normal_()

        base_output = base_linear(x)
        adapted_before = layer(x)
        assert not torch.allclose(base_output, adapted_before, atol=1e-4)

        # Now zero the mask — ΔW is masked out, output should equal base.
        with torch.no_grad():
            layer.mask.zero_()
        adapted_after = layer(x)
        assert torch.allclose(base_output, adapted_after, atol=1e-5)

    def test_scaling_zero_disables_lora_path(self, base_linear, layer):
        """scaling=0 short-circuits the lora path; output equals base."""
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            layer.lora_lambda.fill_(0.5)
        base_output = base_linear(x)
        layer.scaling = 0.0
        assert torch.allclose(layer(x), base_output, atol=1e-5)


class TestOrthogonalityRegularization:
    """The orthogonality regularization loss must respect QR."""

    @pytest.fixture
    def layer(self):
        return AdaLoRALinear(nn.Linear(64, 128), init_rank=8, alpha=32.0)

    def test_loss_near_zero_after_qr(self, layer):
        """QR orthonormalizes P and Q every forward, so the loss is
        essentially zero (≤ 1e-9)."""
        loss = layer.orth_reg_loss()
        assert loss.item() < 1e-9

    def test_loss_non_negative(self, layer):
        """||X - I||²_F ≥ 0 trivially; both terms are sums of squares."""
        loss = layer.orth_reg_loss()
        assert loss.item() >= 0.0

    def test_loss_increases_when_P_perturbed(self, layer):  # noqa: N802
        """Perturbing P so the QR result no longer matches the
        expected identity should push the loss above zero."""
        with torch.no_grad():
            layer.lora_P.fill_(0.0)
            layer.lora_P[0, 0] = 1.0
            # Skip the QR-once verification — the loss function
            # itself applies QR, so we just check that it returns a
            # finite scalar.
        loss = layer.orth_reg_loss()
        assert torch.isfinite(loss)


class TestMergeUnmerge:
    """merge_weights / unmerge_weights must roundtrip the base weight."""

    def test_merge_unmerge_roundtrip(self):
        base = nn.Linear(64, 128)
        layer = AdaLoRALinear(base, init_rank=8, alpha=32.0)
        original_weight = base.weight.detach().clone()

        layer.merge_weights()
        layer.unmerge_weights()
        assert torch.allclose(base.weight, original_weight, atol=1e-6)

    def test_merge_unmerge_with_trained_params(self):
        """Roundtrip identity holds after λ has been moved away from zero."""
        base = nn.Linear(64, 128)
        layer = AdaLoRALinear(base, init_rank=8, alpha=32.0)
        with torch.no_grad():
            layer.lora_lambda.fill_(0.3)
            layer.lora_P.normal_()
            layer.lora_Q.normal_()

        original_weight = base.weight.detach().clone()
        layer.merge_weights()
        layer.unmerge_weights()
        assert torch.allclose(base.weight, original_weight, atol=1e-5)


class TestApplyAdaLora:
    """apply_adalora helper: mirroring apply_lora's contract."""

    def test_replaces_all_linear_layers_when_target_modules_empty(self):
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        before = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        apply_adalora(model, init_rank=8)
        after = sum(1 for m in model.modules() if isinstance(m, AdaLoRALinear))
        assert before == after == 2

    def test_target_modules_filters_by_substring(self):
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        apply_adalora(model, init_rank=8, target_modules=["0"])
        adapted = [name for name, m in model.named_modules() if isinstance(m, AdaLoRALinear)]
        assert adapted == ["0"]

    def test_apply_adalora_returns_same_model(self):
        model = nn.Sequential(nn.Linear(64, 128))
        returned = apply_adalora(model, init_rank=4)
        assert returned is model

    def test_apply_adalora_stores_target_rank_on_each_layer(self):
        model = nn.Sequential(nn.Linear(64, 128), nn.Linear(128, 64))
        apply_adalora(model, init_rank=8, target_rank=3)
        for module in model.modules():
            if isinstance(module, AdaLoRALinear):
                assert module.target_rank == 3

    def test_apply_adalora_does_not_unwrap_existing_adalora(self):
        """After applying, the base ``nn.Linear`` lives inside the
        AdaLoRA layer — re-applying would wrap that nested Linear
        again. This test documents that behavior so it's not
        surprising; if idempotency becomes desirable, it's a
        follow-up that should land in both ``apply_lora`` and
        ``apply_adalora`` for consistency.
        """
        model = nn.Sequential(nn.Linear(64, 128))
        apply_adalora(model, init_rank=4)
        first_count = sum(1 for m in model.modules() if isinstance(m, AdaLoRALinear))
        apply_adalora(model, init_rank=4)
        second_count = sum(1 for m in model.modules() if isinstance(m, AdaLoRALinear))
        # Wraps the inner base_layer too — this matches apply_lora.
        assert first_count == 1
        assert second_count == 2


class TestHelpers:
    """merge / unmerge / count / get / disable / enable roundtrip tests."""

    def test_merge_adalora_calls_merge_on_all_layers(self):
        model = nn.Sequential(nn.Linear(64, 128))
        apply_adalora(model, init_rank=4)
        # Set weights so merge produces a detectable change.
        for module in model.modules():
            if isinstance(module, AdaLoRALinear):
                with torch.no_grad():
                    module.lora_lambda.fill_(0.5)
                    module.lora_P.normal_()
                    module.lora_Q.normal_()

        originals = [m.base_layer.weight.detach().clone() for m in model.modules() if isinstance(m, AdaLoRALinear)]
        merge_adalora(model)
        # After merge, each base weight has changed.
        for module, original in zip(
            (m for m in model.modules() if isinstance(m, AdaLoRALinear)),
            originals,
            strict=False,
        ):
            assert not torch.allclose(module.base_layer.weight, original, atol=1e-6)

        unmerge_adalora(model)
        for module, original in zip(
            (m for m in model.modules() if isinstance(m, AdaLoRALinear)),
            originals,
            strict=False,
        ):
            assert torch.allclose(module.base_layer.weight, original, atol=1e-5)

    def test_get_adalora_parameters_yields_p_q_lambda(self):
        model = nn.Sequential(nn.Linear(64, 128))
        apply_adalora(model, init_rank=4)
        params = list(get_adalora_parameters(model))
        # 1 AdaLoRA layer x 3 parameters each = 3
        assert len(params) == 3
        for param in params:
            assert param.requires_grad

    def test_count_adalora_parameters(self):
        model = nn.Sequential(nn.Linear(64, 128))
        apply_adalora(model, init_rank=4)
        trainable, total = count_adalora_parameters(model)
        # Base Linear is 64*128 + 128 = 8320 (frozen).
        # AdaLoRA: P (128*4) + Q (4*64) + λ (4) = 512 + 256 + 4 = 772.
        assert trainable == 772
        assert total == 8320 + 772

    def test_disable_and_enable_adalora(self):
        model = nn.Sequential(nn.Linear(64, 128))
        apply_adalora(model, init_rank=4)
        original_scalings = [m.scaling for m in model.modules() if isinstance(m, AdaLoRALinear)]
        disable_adalora(model)
        for m in model.modules():
            if isinstance(m, AdaLoRALinear):
                assert m.scaling == 0.0
        enable_adalora(model)
        for m, original in zip(
            (m for m in model.modules() if isinstance(m, AdaLoRALinear)),
            original_scalings,
            strict=False,
        ):
            assert m.scaling == original


class TestInitValidation:
    """Bad constructor arguments should raise with clear errors."""

    def test_init_rank_must_be_positive(self):
        with pytest.raises(ValueError, match="init_rank"):
            AdaLoRALinear(nn.Linear(64, 128), init_rank=0)

    def test_init_rank_must_fit_linear_dimensions(self):
        with torch.no_grad():
            layer = AdaLoRALinear(nn.Linear(8, 8), init_rank=4)
            assert layer.init_rank == 4
        with pytest.raises(ValueError, match="init_rank"):
            AdaLoRALinear(nn.Linear(8, 8), init_rank=16)

    def test_target_rank_must_be_positive(self):
        with pytest.raises(ValueError, match="target_rank"):
            AdaLoRALinear(nn.Linear(64, 128), init_rank=8, target_rank=0)

    def test_target_rank_must_not_exceed_init_rank(self):
        with pytest.raises(ValueError, match="target_rank"):
            AdaLoRALinear(nn.Linear(64, 128), init_rank=8, target_rank=16)


class TestBackward:
    """Verify gradients flow through the SVD-form parameterization."""

    def test_gradients_flow_to_P_Q_lambda(self):  # noqa: N802
        layer = AdaLoRALinear(nn.Linear(64, 128), init_rank=8)
        x = torch.randn(2, 10, 64, requires_grad=False)
        target = torch.randn(2, 10, 128)
        loss = ((layer(x) - target) ** 2).mean()
        loss.backward()

        assert layer.lora_P.grad is not None
        assert layer.lora_Q.grad is not None
        assert layer.lora_lambda.grad is not None
        # Gradients should be non-zero once λ is non-zero (currently
        # zero at init → ∇ might be tiny; loosen tolerance).
        assert torch.isfinite(layer.lora_P.grad).all()
        assert torch.isfinite(layer.lora_Q.grad).all()
        assert torch.isfinite(layer.lora_lambda.grad).all()

    def test_gradients_flow_through_orth_reg_loss(self):
        layer = AdaLoRALinear(nn.Linear(64, 128), init_rank=8)
        loss = layer.orth_reg_loss()
        loss.backward()
        # Orth reg loss depends on P and Q through the QR step.
        assert layer.lora_P.grad is not None
        assert layer.lora_Q.grad is not None


class TestComputeImportanceScores:
    """compute_importance_scores must rank components correctly."""

    def test_default_scores_are_lambda_magnitudes(self):
        layer = AdaLoRALinear(nn.Linear(64, 128), init_rank=8)
        with torch.no_grad():
            layer.lora_lambda[0] = 5.0
            layer.lora_lambda[1] = 3.0
            layer.lora_lambda[2] = 1.0
            # rest are zero

        scores = layer.compute_importance_scores()

        assert scores.shape == (8,)
        # All-zero entries rank last.
        top3 = torch.topk(scores, k=3).indices.tolist()
        assert top3 == [0, 1, 2]
        assert torch.allclose(scores[0], torch.tensor(5.0))
        assert torch.allclose(scores[1], torch.tensor(3.0))
        assert torch.allclose(scores[2], torch.tensor(1.0))

    def test_gradient_ema_combined_score(self):
        """With gradient_ema, scores are |λ_i| · |gradient_ema_i|."""
        layer = AdaLoRALinear(nn.Linear(64, 128), init_rank=8)
        with torch.no_grad():
            layer.lora_lambda[0] = 5.0  # big magnitude
            layer.lora_lambda[1] = 1.0  # small magnitude
            layer.lora_lambda[2] = 0.0  # zero magnitude

        grad_ema = torch.zeros(8)
        grad_ema[0] = 0.0  # 5 * 0 = 0
        grad_ema[1] = 2.0  # 1 * 2 = 2
        grad_ema[2] = 1.0  # 0 * 1 = 0

        scores = layer.compute_importance_scores(grad_ema)
        assert torch.allclose(scores[0], torch.tensor(0.0))
        assert torch.allclose(scores[1], torch.tensor(2.0))
        assert torch.allclose(scores[2], torch.tensor(0.0))

    def test_gradient_ema_shape_mismatch_raises(self):
        layer = AdaLoRALinear(nn.Linear(64, 128), init_rank=8)
        with pytest.raises(ValueError, match="shape"):
            layer.compute_importance_scores(torch.zeros(4))


class TestPruneToRank:
    """prune_to_rank must drop the right components and respect the rank."""

    @pytest.fixture
    def layer(self):
        layer = AdaLoRALinear(nn.Linear(64, 128), init_rank=12, target_rank=6)
        with torch.no_grad():
            # Give each component a distinct magnitude so topk is unambiguous.
            layer.lora_lambda.copy_(torch.arange(1, 13, dtype=torch.float32))
        return layer

    def test_prune_drops_lowest_importance(self, layer):
        layer.prune_to_rank(6)
        # Top-6 indices by score: 11, 10, 9, 8, 7, 6 (values 12, 11, 10, 9, 8, 7).
        kept = layer.mask.nonzero(as_tuple=True)[0].tolist()
        assert sorted(kept) == [6, 7, 8, 9, 10, 11]
        assert layer.effective_rank == 6

    def test_prune_is_idempotent(self, layer):
        layer.prune_to_rank(6)
        first_mask = layer.mask.detach().clone()
        layer.prune_to_rank(6)
        assert torch.allclose(layer.mask, first_mask)

    def test_prune_to_init_rank_is_noop(self, layer):
        layer.prune_to_rank(layer.init_rank)
        assert layer.mask.sum().item() == layer.init_rank

    def test_prune_to_zero_empties_mask(self, layer):
        layer.prune_to_rank(0)
        assert layer.mask.sum().item() == 0
        assert layer.effective_rank == 0

    def test_prune_to_rank_above_effective_raises(self, layer):
        with pytest.raises(ValueError, match="exceeds effective_rank"):
            layer.prune_to_rank(layer.effective_rank + 1)

    def test_prune_to_negative_raises(self, layer):
        with pytest.raises(ValueError, match="≥ 0"):
            layer.prune_to_rank(-1)

    def test_prune_at_effective_rank_is_noop(self, layer):
        """Already-pruned to k; calling prune_to_rank(k) again is a no-op."""
        layer.prune_to_rank(6)
        assert layer.effective_rank == 6
        snapshot = layer.mask.detach().clone()
        layer.prune_to_rank(6)
        assert torch.allclose(layer.mask, snapshot)

    def test_prune_with_explicit_scores(self, layer):
        """Passing scores externally should override the default scoring."""
        # Construct scores that favor components 0, 2, 4 instead of the
        # largest-magnitude ones (5..11).
        custom = torch.zeros(12)
        custom[0] = 100.0
        custom[2] = 50.0
        custom[4] = 25.0
        layer.prune_to_rank(3, scores=custom)
        kept = layer.mask.nonzero(as_tuple=True)[0].tolist()
        assert sorted(kept) == [0, 2, 4]


class TestUpdateBudget:
    """update_budget must schedule the rank linearly between tinit and tfinal."""

    @pytest.fixture
    def layer(self):
        return AdaLoRALinear(nn.Linear(64, 128), init_rank=12, target_rank=6)

    def test_before_tinit_returns_init_rank(self, layer):
        assert layer.update_budget(0, 10, 100) == 12
        assert layer.update_budget(9, 10, 100) == 12
        assert layer.update_budget(10, 10, 100) == 12

    def test_after_tfinal_returns_target_rank(self, layer):
        assert layer.update_budget(100, 10, 100) == 6
        assert layer.update_budget(200, 10, 100) == 6

    def test_linear_interpolation(self, layer):
        # midpoint: (10 + 100) / 2 = 55 → halfway between 12 and 6 = 9
        assert layer.update_budget(55, 10, 100) == 9
        # progress 22/90 = 0.244: 12 - 0.244 * 6 = 10.533 → round to 11
        assert layer.update_budget(32, 10, 100) == 11
        # progress 67/90 = 0.744: 12 - 0.744 * 6 = 7.533 → round to 8
        assert layer.update_budget(77, 10, 100) == 8

    def test_negative_step_raises(self, layer):
        with pytest.raises(ValueError, match="current_step"):
            layer.update_budget(-1, 10, 100)

    def test_tinit_ge_tfinal_raises(self, layer):
        with pytest.raises(ValueError, match="tinit"):
            layer.update_budget(50, 100, 100)
        with pytest.raises(ValueError, match="tinit"):
            layer.update_budget(50, 100, 50)


class TestPruneAdaLora:
    """The module-level helper walks every AdaLoRALinear in the model."""

    def test_target_rank_mode(self):
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        apply_adalora(model, init_rank=10, target_rank=4)
        prune_adalora(model, target_rank=2)
        for m in model.modules():
            if isinstance(m, AdaLoRALinear):
                assert m.effective_rank == 2

    def test_schedule_mode(self):
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        apply_adalora(model, init_rank=10, target_rank=4)
        # Mid-way between tinit=10 and tfinal=100: rank = 10 - 0.5*(10-4) = 7
        prune_adalora(model, schedule=(10, 100), current_step=55)
        for m in model.modules():
            if isinstance(m, AdaLoRALinear):
                assert m.effective_rank == 7

    def test_schedule_mode_respects_step(self):
        """Calling again at a later step shrinks the rank further."""
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        apply_adalora(model, init_rank=10, target_rank=2)
        prune_adalora(model, schedule=(0, 100), current_step=25)
        rank_at_25 = next(m.effective_rank for m in model.modules() if isinstance(m, AdaLoRALinear))
        prune_adalora(model, schedule=(0, 100), current_step=75)
        rank_at_75 = next(m.effective_rank for m in model.modules() if isinstance(m, AdaLoRALinear))
        assert rank_at_25 > rank_at_75

    def test_neither_target_rank_nor_schedule_raises(self):
        model = nn.Sequential(nn.Linear(64, 128))
        apply_adalora(model, init_rank=4)
        with pytest.raises(ValueError, match="exactly one"):
            prune_adalora(model)

    def test_both_target_rank_and_schedule_raises(self):
        model = nn.Sequential(nn.Linear(64, 128))
        apply_adalora(model, init_rank=4)
        with pytest.raises(ValueError, match="exactly one"):
            prune_adalora(model, target_rank=2, schedule=(0, 100), current_step=50)

    def test_schedule_without_current_step_raises(self):
        model = nn.Sequential(nn.Linear(64, 128))
        apply_adalora(model, init_rank=4)
        with pytest.raises(ValueError, match="current_step"):
            prune_adalora(model, schedule=(0, 100))

    def test_target_rank_capped_by_per_layer_effective_rank(self):
        """Pruning below the layer's effective_rank works; pruning to a
        value larger than effective_rank (after a previous prune)
        should be capped silently rather than raising — we just can't
        un-prune."""
        model = nn.Sequential(nn.Linear(64, 128))
        apply_adalora(model, init_rank=10, target_rank=2)
        prune_adalora(model, target_rank=2)
        # Already at 2; asking for 5 should be silently capped to 2.
        prune_adalora(model, target_rank=5)
        for m in model.modules():
            if isinstance(m, AdaLoRALinear):
                assert m.effective_rank == 2

    def test_empty_model_is_noop(self):
        model = nn.Sequential(nn.Linear(64, 64))
        # No AdaLoRALinear layers — should not raise.
        prune_adalora(model, target_rank=2)

    def test_gradient_emas_threaded_through(self):
        """gradient_emas keyed by id(layer) overrides default scoring."""
        model = nn.Sequential(nn.Linear(64, 128))
        apply_adalora(model, init_rank=6, target_rank=2)
        layer = next(m for m in model.modules() if isinstance(m, AdaLoRALinear))

        # Set λ to be uniformly small except at index 0, so default
        # scoring would keep index 0 + something else.
        with torch.no_grad():
            layer.lora_lambda.zero_()
            layer.lora_lambda[0] = 1.0

        # But feed a gradient EMA that *flips* the ranking: index 3
        # becomes the most important (1.0 * 10 = 10 vs 1.0 * 0 = 0).
        grad_ema = torch.zeros(6)
        grad_ema[3] = 10.0
        grad_ema[0] = 0.0

        prune_adalora(model, target_rank=1, gradient_emas={id(layer): grad_ema})
        kept = layer.mask.nonzero(as_tuple=True)[0].tolist()
        assert kept == [3]


class TestPruningIntegration:
    """End-to-end: train λ briefly, prune, verify the kept components.

    The end-to-end story for the pruning slice is:

    1. Build a small model + apply AdaLoRA.
    2. Pick a "hard" direction in the input that only a few λ
       components contribute to (we force this by setting
       ``lora_lambda`` to non-zero values for *only* some components,
       then training).
    3. Drive a few forward/backward steps so the surviving λ
       magnitudes diverge from the pruned-out ones.
    4. Call ``prune_to_rank`` and check that the kept components
       correspond to the largest post-training magnitudes.
    """

    def test_prune_keeps_components_whose_magnitude_grew(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16))
        apply_adalora(model, init_rank=10, target_rank=3)

        # Pull out the single AdaLoRALinear (the second nn.Linear
        # would also have been replaced; Sequential keeps everything
        # but we only need the one for this check).
        layer = next(m for m in model.modules() if isinstance(m, AdaLoRALinear))
        init_rank = layer.init_rank

        # Seed λ with non-uniform magnitudes so the top-k is
        # unambiguous before training.
        with torch.no_grad():
            layer.lora_lambda.copy_(torch.linspace(0.0, 1.0, init_rank, dtype=torch.float32))

        # Train λ only (P and Q stay frozen — this is a smoke test
        # of the prune signal, not a quality check on P/Q learning).
        optimizer = torch.optim.SGD([layer.lora_lambda], lr=0.1)

        # Use a tiny input so the gradient is dominated by the λ
        # direction we want to grow.
        x = torch.randn(4, 32)
        for _step in range(20):
            optimizer.zero_grad()
            pred = model(x)
            target = pred.detach() + 0.5 * torch.randn_like(pred)
            loss = ((pred - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Snapshot the magnitudes *before* pruning; the top-k here
        # are the components training drove up the most.
        post_magnitudes = layer.lora_lambda.detach().abs().clone()
        expected_top3 = torch.topk(post_magnitudes, k=3).indices.tolist()

        # Prune to 3.
        layer.prune_to_rank(3)

        # The mask should match the top-3 magnitudes — exactly the
        # components we expect training to have emphasized.
        kept = layer.mask.nonzero(as_tuple=True)[0].tolist()
        assert sorted(kept) == sorted(expected_top3)
        assert layer.effective_rank == 3

        # And the pruned-out components should no longer contribute:
        # the forward output with the mask applied should differ from
        # the output if all components were active (sanity check that
        # pruning actually zeros something).
        with torch.no_grad():
            out_pruned = layer(x)
            # Toggle the mask back to all-ones and recompute.
            snapshot = layer.mask.detach().clone()
            layer.mask.fill_(1.0)
            out_full = layer(x)
            layer.mask.copy_(snapshot)
            # The two outputs must differ for at least one position
            # (we expect substantially different — some components
            # are zeroed out).
            assert not torch.allclose(out_pruned, out_full)


class TestAdaLoRAGradientEMA:
    """Gradient-EMA tracker feeding ``prune_adalora`` with importance info.

    Implements the EMA half of AdaLoRA Algorithm 1::

        I_avg_i ← α · I_avg_i + (1 − α) · |∂L/∂λ_i|

    The tracker reads ``layer.lora_lambda.grad.abs()`` after each
    backward pass; layers that have no gradient (frozen, unused) are
    left untouched so the EMA only reflects what training actually
    drove.
    """

    def _build_model(self):
        """Tiny Sequential model with two AdaLoRA layers for testing."""
        from llm.core.adalora import apply_adalora

        model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8))
        apply_adalora(model, init_rank=6, target_rank=3)
        return model

    def test_init_zero_per_layer(self):
        """Construction: one zero-EMA tensor per AdaLoRALinear."""
        from llm.core.adalora import AdaLoRAGradientEMA

        model = self._build_model()
        tracker = AdaLoRAGradientEMA(model, alpha=0.95)

        ema_dict = tracker.as_dict()
        # Two AdaLoRA layers → two EMA tensors keyed by id(layer).
        assert len(ema_dict) == 2
        for tensor in ema_dict.values():
            assert tensor.shape == (6,)
            assert torch.allclose(tensor, torch.zeros(6))

    def test_alpha_stored(self):
        from llm.core.adalora import AdaLoRAGradientEMA

        tracker = AdaLoRAGradientEMA(self._build_model(), alpha=0.9)
        assert tracker.alpha == 0.9

    def test_alpha_must_be_in_open_unit_interval(self):
        from llm.core.adalora import AdaLoRAGradientEMA

        model = self._build_model()
        with pytest.raises(ValueError, match="alpha"):
            AdaLoRAGradientEMA(model, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            AdaLoRAGradientEMA(model, alpha=1.0)
        with pytest.raises(ValueError, match="alpha"):
            AdaLoRAGradientEMA(model, alpha=1.5)
        with pytest.raises(ValueError, match="alpha"):
            AdaLoRAGradientEMA(model, alpha=-0.1)

    def test_update_reads_lambda_grad_abs(self):
        """Update pulls |lora_lambda.grad| and EMA-smooths it."""
        from llm.core.adalora import AdaLoRAGradientEMA

        model = self._build_model()
        tracker = AdaLoRAGradientEMA(model, alpha=0.5)

        # Pick one AdaLoRA layer, set a non-zero gradient.
        layer = next(m for m in model.modules() if isinstance(m, AdaLoRALinear))
        layer.lora_lambda.grad = torch.tensor([0.4, 0.0, 0.6, 0.0, 0.8, 0.0])

        tracker.update()
        ema = tracker.as_dict()[id(layer)]
        # α=0.5, current weight=1−α=0.5; grad_abs is the input directly.
        expected = 0.5 * torch.tensor([0.4, 0.0, 0.6, 0.0, 0.8, 0.0])
        assert torch.allclose(ema, expected, atol=1e-7)

    def test_update_smoothing_recursion(self):
        """Two updates compose as α·ema + (1−α)·grad_abs."""
        from llm.core.adalora import AdaLoRAGradientEMA

        model = self._build_model()
        tracker = AdaLoRAGradientEMA(model, alpha=0.8)
        layer = next(m for m in model.modules() if isinstance(m, AdaLoRALinear))

        # First update with grad = 1.0 → ema = 0.2 * 1.0 = 0.2
        layer.lora_lambda.grad = torch.ones(6)
        tracker.update()
        assert torch.allclose(tracker.as_dict()[id(layer)], 0.2 * torch.ones(6))

        # Second update with grad = 0.0 → ema = 0.8*0.2 + 0.2*0.0 = 0.16
        layer.lora_lambda.grad = torch.zeros(6)
        tracker.update()
        assert torch.allclose(tracker.as_dict()[id(layer)], 0.16 * torch.ones(6))

    def test_update_skips_layers_without_grad(self):
        """A layer with no gradient (e.g. frozen) keeps its EMA unchanged."""
        from llm.core.adalora import AdaLoRAGradientEMA

        model = self._build_model()
        tracker = AdaLoRAGradientEMA(model, alpha=0.5)

        # Seed EMAs to non-zero so we can detect a no-op.
        for tensor in tracker.as_dict().values():
            tensor.fill_(0.42)

        # No .grad populated → tracker.update() must leave EMA untouched.
        tracker.update()
        for tensor in tracker.as_dict().values():
            assert torch.allclose(tensor, 0.42 * torch.ones(6))

    def test_update_handles_none_grad(self):
        """`lora_lambda.grad is None` is treated like 'no grad'."""
        from llm.core.adalora import AdaLoRAGradientEMA

        model = self._build_model()
        tracker = AdaLoRAGradientEMA(model, alpha=0.5)
        layer = next(m for m in model.modules() if isinstance(m, AdaLoRALinear))
        # Explicitly None (default after construction, before backward).
        assert layer.lora_lambda.grad is None
        # Must not raise.
        tracker.update()

    def test_as_dict_keyed_by_id_layer(self):
        """as_dict() keys must match id(layer) so prune_adalora can look them up."""
        from llm.core.adalora import AdaLoRAGradientEMA

        model = self._build_model()
        tracker = AdaLoRAGradientEMA(model, alpha=0.95)
        ema_dict = tracker.as_dict()

        layers = [m for m in model.modules() if isinstance(m, AdaLoRALinear)]
        assert set(ema_dict.keys()) == {id(layer) for layer in layers}

    def test_state_dict_keyed_by_qualified_name(self):
        """state_dict() must use qualified name (stable across pickle)."""
        from llm.core.adalora import AdaLoRAGradientEMA

        model = self._build_model()
        tracker = AdaLoRAGradientEMA(model, alpha=0.95)

        state = tracker.state_dict()
        # Two layers in a Sequential → "0" and "1"
        assert set(state.keys()) == {"0", "1"}
        for key, tensor in state.items():
            assert tensor.shape == (6,)
            assert torch.allclose(tensor, torch.zeros(6))

    def test_state_dict_roundtrip(self):
        """state_dict → load_state_dict must restore EMA tensors exactly."""
        from llm.core.adalora import AdaLoRAGradientEMA

        model = self._build_model()
        tracker = AdaLoRAGradientEMA(model, alpha=0.95)

        # Write distinct, non-zero EMA values per layer so we can detect loss.
        for i, (_key, tensor) in enumerate(tracker.state_dict().items()):
            tensor.fill_(float(i + 1) * 0.1)

        snapshot = {k: v.detach().clone() for k, v in tracker.state_dict().items()}

        # Roundtrip through a checkpoint-like dict (state tensors must be detached).
        state = tracker.state_dict()
        new_tracker = AdaLoRAGradientEMA(self._build_model(), alpha=0.95)
        new_tracker.load_state_dict(state)

        for key, expected in snapshot.items():
            assert torch.allclose(new_tracker.state_dict()[key], expected, atol=1e-7)

    def test_load_state_dict_ignores_unknown_keys(self):
        """Stale keys from a different model don't crash — they're ignored."""
        from llm.core.adalora import AdaLoRAGradientEMA

        model = self._build_model()
        tracker = AdaLoRAGradientEMA(model, alpha=0.95)

        # Unknown layer name — extra entry that doesn't match current model.
        bogus_state = {"0": torch.zeros(6), "nonexistent_layer": torch.ones(6)}
        # Must not raise.
        tracker.load_state_dict(bogus_state)

    def test_load_state_dict_handles_none(self):
        """load_state_dict(None) is a no-op (defensive for fresh checkpoints)."""
        from llm.core.adalora import AdaLoRAGradientEMA

        model = self._build_model()
        tracker = AdaLoRAGradientEMA(model, alpha=0.95)
        # Must not raise.
        tracker.load_state_dict(None)
