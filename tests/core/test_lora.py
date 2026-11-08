"""
Tests for LoRA (Low-Rank Adaptation) module.
"""

import pytest
import torch
import torch.nn as nn

from llm.core.lora import (
    LoRALinear,
    apply_lora,
    count_lora_parameters,
    disable_lora,
    enable_lora,
    get_lora_parameters,
    merge_lora,
    unmerge_lora,
)
from llm.models.decoder import DecoderModel


class TestLoRALinear:
    """Tests for LoRALinear class."""

    @pytest.fixture
    def base_linear(self):
        return nn.Linear(64, 128)

    @pytest.fixture
    def lora_layer(self, base_linear):
        return LoRALinear(base_linear, rank=8, alpha=16.0)

    def test_init(self, lora_layer):
        """Test LoRALinear initialization."""
        assert lora_layer.rank == 8
        assert lora_layer.alpha == 16.0
        assert lora_layer.scaling == 2.0  # 16 / 8

        # Check shapes
        assert lora_layer.lora_A.shape == (64, 8)
        assert lora_layer.lora_B.shape == (8, 128)

    def test_base_layer_frozen(self, lora_layer):
        """Test that base layer weights are frozen."""
        assert not lora_layer.base_layer.weight.requires_grad
        if lora_layer.base_layer.bias is not None:
            assert not lora_layer.base_layer.bias.requires_grad

    def test_lora_params_trainable(self, lora_layer):
        """Test that LoRA parameters are trainable."""
        assert lora_layer.lora_A.requires_grad
        assert lora_layer.lora_B.requires_grad

    def test_forward_initial(self, base_linear, lora_layer):
        """Test that initial output matches base layer (B is zero-initialized)."""
        x = torch.randn(2, 10, 64)
        base_output = base_linear(x)
        lora_output = lora_layer(x)

        # Should be identical since B is initialized to zeros
        assert torch.allclose(base_output, lora_output, atol=1e-6)

    def test_forward_after_training(self, lora_layer):
        """Test forward pass after modifying LoRA weights."""
        x = torch.randn(2, 10, 64)

        # Modify B to non-zero
        lora_layer.lora_B.data.fill_(0.1)

        output = lora_layer(x)
        base_output = lora_layer.base_layer(x)

        # Output should differ from base
        assert not torch.allclose(base_output, output)

    def test_trainable_parameters_count(self, lora_layer):
        """Test trainable parameter count."""
        expected = 64 * 8 + 8 * 128  # A + B
        assert lora_layer.trainable_parameters == expected

    def test_merge_weights_is_output_preserving(self, lora_layer):
        """Merging fold the adapter into the base *without* double-counting.

        A correct merge must be semantically invisible: ``merge_weights()``
        folds the delta into ``base_layer.weight`` and disables the adapter
        path, so ``forward()`` (through the wrapper — not just a hand-picked
        ``base_layer(x)`` call) continues to produce the pre-merge output.
        Otherwise the merged model double-applies the adapter at serve time.
        """
        x = torch.randn(2, 10, 64)
        lora_layer.lora_B.data.fill_(0.1)

        output_before = lora_layer(x).clone()

        lora_layer.merge_weights()
        output_merged = lora_layer(x)

        assert torch.allclose(output_before, output_merged, atol=1e-5)

    def test_unmerge_weights(self, lora_layer):
        """Test weight unmerging."""
        original_weight = lora_layer.base_layer.weight.clone()
        lora_layer.lora_B.data.fill_(0.1)

        lora_layer.merge_weights()
        lora_layer.unmerge_weights()

        assert torch.allclose(lora_layer.base_layer.weight, original_weight, atol=1e-5)
        # unmerge also restores the adapter path (scaling no longer 0)
        assert lora_layer.scaling == lora_layer.alpha / lora_layer.rank

    def test_double_merge_is_idempotent_then_unmerge_restores(self, lora_layer):
        """Regression: calling ``merge_weights`` twice (a
        defensive/loop caller, or serve-merge after an earlier export-merge)
        must NOT clobber the unmerge snapshot. The old code re-stored
        ``_merged_scaling = self.scaling`` on the second call — but scaling
        is already 0 post-first-merge, so the re-fold no-ops AND the
        snapshot becomes 0; the later ``unmerge_weights`` then restores
        scaling=0 and leaves the base permanently folded at 2x adapter
        strength with no error raised. A second merge must be a no-op."""
        original_weight = lora_layer.base_layer.weight.clone()
        scaling = lora_layer.scaling
        lora_layer.lora_B.data.fill_(0.1)

        lora_layer.merge_weights()
        lora_layer.merge_weights()  # second merge must not clobber the snapshot
        lora_layer.unmerge_weights()

        assert torch.allclose(lora_layer.base_layer.weight, original_weight, atol=1e-5), (
            "double-merge + unmerge must roundtrip the base weight"
        )
        assert lora_layer.scaling == scaling

    def test_merge_then_forward_equals_base_after_disable(self, lora_layer):
        """Merged layer's forward matches its folded base exactly.

        This catches a regression where merge left the lora path active,
        producing ``base + 2 * adapter`` (double-count). After merge the
        wrapper must be a pure pass-through to the folded base.
        """
        x = torch.randn(2, 10, 64)
        lora_layer.lora_B.data.fill_(0.1)

        lora_layer.merge_weights()

        assert torch.allclose(lora_layer(x), lora_layer.base_layer(x), atol=1e-6)


class TestApplyLoRA:
    """Tests for apply_lora function."""

    def test_apply_to_all_linear(self):
        """Test applying LoRA to all linear layers."""
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

        apply_lora(model, rank=4, alpha=8.0)

        # Count LoRA layers
        lora_count = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
        assert lora_count == 3

    def test_apply_to_specific_modules(self):
        """Test applying LoRA to specific modules only."""
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

        # Only apply to first linear layer (named "0")
        apply_lora(model, rank=4, target_modules=["0"])

        lora_count = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
        assert lora_count == 1

    def test_apply_to_decoder_model(self):
        """Test applying LoRA to DecoderModel."""
        model = DecoderModel(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            max_seq_len=32,
        )

        # Apply to attention QKV and output projections
        apply_lora(model, rank=8, target_modules=["qkv_proj", "out_proj"])

        lora_count = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
        assert lora_count > 0


class TestLoRAUtilities:
    """Tests for LoRA utility functions."""

    @pytest.fixture
    def model_with_lora(self):
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        apply_lora(model, rank=4, alpha=8.0)
        return model

    def test_get_lora_parameters(self, model_with_lora):
        """Test getting LoRA parameters."""
        lora_params = list(get_lora_parameters(model_with_lora))
        # 2 linear layers * 2 params (A, B) each
        assert len(lora_params) == 4

    def test_count_lora_parameters(self, model_with_lora):
        """Test counting trainable vs total parameters."""
        trainable, total = count_lora_parameters(model_with_lora)

        # Trainable should be much less than total
        assert trainable < total
        assert trainable > 0

    def test_disable_enable_lora(self, model_with_lora):
        """Test disabling and re-enabling LoRA."""
        x = torch.randn(2, 32)

        # Modify B weights
        for m in model_with_lora.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.data.fill_(0.1)

        output_enabled = model_with_lora(x).clone()

        disable_lora(model_with_lora)
        output_disabled = model_with_lora(x).clone()

        enable_lora(model_with_lora)
        output_reenabled = model_with_lora(x)

        # Disabled output should differ from enabled
        assert not torch.allclose(output_enabled, output_disabled)
        # Re-enabled should match original
        assert torch.allclose(output_enabled, output_reenabled)

    def test_merge_unmerge_full_model(self, model_with_lora):
        """Test merge and unmerge on full model."""
        x = torch.randn(2, 32)

        for m in model_with_lora.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.data.fill_(0.1)

        output_before = model_with_lora(x).clone()

        merge_lora(model_with_lora)
        unmerge_lora(model_with_lora)

        output_after = model_with_lora(x)

        assert torch.allclose(output_before, output_after, atol=1e-5)


class TestLoRATraining:
    """Tests for training with LoRA."""

    def test_training_only_lora_params(self):
        """Test that only LoRA parameters are updated during training."""
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        apply_lora(model, rank=4)

        # Store original base weights
        original_weights = {}
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                original_weights[name] = module.base_layer.weight.clone()

        # Train
        optimizer = torch.optim.Adam(get_lora_parameters(model), lr=0.01)
        x = torch.randn(4, 32)
        target = torch.randn(4, 10)
        criterion = nn.MSELoss()

        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Verify base weights unchanged
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                assert torch.allclose(module.base_layer.weight, original_weights[name])

    def test_decoder_model_training_with_lora(self, device):
        """Test training DecoderModel with LoRA."""
        model = DecoderModel(
            vocab_size=100,
            hidden_size=64,
            num_layers=1,
            num_heads=2,
            max_seq_len=32,
        ).to(device)

        # Apply LoRA
        apply_lora(model, rank=4, target_modules=["qkv_proj", "out_proj", "lm_head"])

        trainable, total = count_lora_parameters(model)
        reduction = 1 - (trainable / total)
        print(f"Parameter reduction: {reduction:.1%}")

        # Train
        optimizer = torch.optim.Adam(get_lora_parameters(model), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        input_ids = torch.randint(0, 100, (2, 16), device=device)
        labels = torch.randint(0, 100, (2, 16), device=device)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, 100), labels.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Verify training happened
        assert losses[-1] != losses[0]


# --- disable/enable must survive the merge state and repeat calls ---


class TestDisableEnableInteractions:
    """``disable_lora``/``enable_lora`` interact with the merge snapshot.

    Both write ``scaling``, which is the single value ``forward`` uses to decide
    whether the adapter contributes. Recording a wrong value here is silent: the
    model runs, it just is (or is not) adapted.
    """

    @staticmethod
    def _layer():
        torch.manual_seed(0)
        layer = LoRALinear(nn.Linear(16, 16))
        with torch.no_grad():
            layer.lora_B.normal_(0, 0.1)
        return layer

    def test_disable_then_enable_restores_scaling(self):
        layer = self._layer()
        original = layer.scaling
        disable_lora(nn.Sequential(layer))
        assert layer.scaling == 0.0
        enable_lora(nn.Sequential(layer))
        assert layer.scaling == original

    def test_disable_twice_keeps_the_first_snapshot(self):
        """``disable`` records ``scaling``; re-recording would store 0 and lose it."""
        model = nn.Sequential(self._layer())
        layer = model[0]
        original = layer.scaling

        disable_lora(model)
        disable_lora(model)  # e.g. a looping/defensive caller
        enable_lora(model)

        assert layer.scaling == original, "a second disable overwrote the snapshot with 0"

    def test_disabling_a_merged_layer_does_not_erase_the_adapter(self):
        """A merged layer is already inert; disable must not clobber its snapshot.

        ``merge_weights`` sets ``scaling = 0`` and stores the real value in
        ``_merged_scaling``. Recording that 0 in ``_original_scaling`` made a
        later ``enable`` restore 0 — the adapter was gone for good with no error.
        """
        model = nn.Sequential(self._layer())
        layer = model[0]
        original = layer.scaling

        merge_lora(model)
        assert layer.scaling == 0.0
        disable_lora(model)
        enable_lora(model)

        # Assert BEFORE unmerging: unmerge restores scaling from its own
        # _merged_scaling snapshot, which would mask a clobbered adapter here.
        # While merged the layer is a pass-through by design, so enable() must
        # NOT reactivate the adapter path (that would double-count the delta).
        assert layer.scaling == 0.0, "enable reactivated the adapter on a merged layer"

        unmerge_lora(model)
        assert layer.scaling == original, "the adapter was lost across merge -> disable -> enable -> unmerge"


class TestLoRAGradientRouting:
    """Which parameters a backward reaches, and with what magnitude.

    `test_lora.py` had no gradient test at all: every case checks forward outputs
    or attribute values. That leaves the training path — the reason LoRA exists —
    unpinned. Two properties matter and both are cheap to check.
    """

    @staticmethod
    def _layer(rank=4, alpha=8.0, seed=0):
        import torch.nn as nn

        from llm.core.lora import LoRALinear

        torch.manual_seed(seed)
        base = nn.Linear(8, 6)
        layer = LoRALinear(base, rank=rank, alpha=alpha)
        with torch.no_grad():
            layer.lora_B.normal_(0, 0.5)  # B is zero-initialised; give it signal
        return layer, base

    def test_a_backward_populates_only_the_adapter(self):
        """The base weight is frozen, so only lora_A / lora_B may receive grads."""
        torch.manual_seed(0)
        layer, base = self._layer()
        x = torch.randn(3, 8)

        layer(x).sum().backward()

        assert layer.lora_A.grad is not None, "lora_A received no gradient"
        assert layer.lora_B.grad is not None, "lora_B received no gradient"
        assert layer.lora_A.grad.abs().sum() > 0, "lora_A gradient is all zeros"
        assert layer.lora_B.grad.abs().sum() > 0, "lora_B gradient is all zeros"
        # The base layer is frozen in a LoRA adapter — no gradient may reach it.
        assert base.weight.grad is None, "a gradient reached the frozen base weight"
        if base.bias is not None:
            assert base.bias.grad is None, "a gradient reached the frozen base bias"

    def test_merge_disables_the_adapter_gradient(self):
        """After merge the branch is folded in (scaling 0), so its grads must vanish.

        This is the deploy-side invariant: a merged layer is meant to be served, not
        fine-tuned further. If grads kept flowing, an optimizer step after merge
        would move weights that the folded base already accounts for — silently
        double-counting the update.
        """
        torch.manual_seed(0)
        layer, _ = self._layer()
        x = torch.randn(3, 8)

        layer.merge_weights()
        assert layer.scaling == 0.0, "merge did not zero the adapter scaling"

        layer(x).sum().backward()

        assert layer.lora_A.grad is not None, "lora_A has no gradient recorded after merge"
        assert layer.lora_B.grad is not None, "lora_B has no gradient recorded after merge"
        assert layer.lora_A.grad.abs().max() == 0, (
            f"lora_A still receives gradient after merge: {layer.lora_A.grad.abs().max()}"
        )
        assert layer.lora_B.grad.abs().max() == 0, (
            f"lora_B still receives gradient after merge: {layer.lora_B.grad.abs().max()}"
        )

    def test_disable_then_enable_restores_the_forward(self):
        """`disable_lora` must make forward exactly equal the base layer."""
        import torch.nn.functional as functional

        torch.manual_seed(0)
        layer, base = self._layer()
        x = torch.randn(3, 8)

        from llm.core.lora import disable_lora, enable_lora

        with torch.no_grad():
            adapted = layer(x).clone()
        disable_lora(layer)
        with torch.no_grad():
            disabled = layer(x)
        enable_lora(layer)
        with torch.no_grad():
            reenabled = layer(x)

        assert not torch.allclose(adapted, functional.linear(x, base.weight, base.bias)), (
            "the adapter had no effect before disabling — the test cannot detect a broken disable"
        )
        assert torch.allclose(disabled, functional.linear(x, base.weight, base.bias), atol=1e-6), (
            "disabled forward does not equal the base layer"
        )
        assert torch.allclose(reenabled, adapted, atol=1e-6), "re-enabling did not restore the adapter"
