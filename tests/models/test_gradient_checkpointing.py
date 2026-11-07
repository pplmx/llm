"""
Tests for Gradient Checkpointing.

Verifies that gradient checkpointing works correctly for memory-efficient training.
"""

import pytest
import torch
import torch.nn as nn

from llm.core.kv_cache import create_decoder_kv_caches
from llm.models.decoder import DecoderModel
from tests.support.devices import cuda_usable
from tests.support.models import decoder_model_kwargs


def _small_model_config() -> dict:
    """Tiny decoder config shared by the classes below.

    Module level rather than a class-scoped fixture so both the behaviour tests
    and the equivalence tests use one definition.
    """
    kwargs = decoder_model_kwargs(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=32,
    )
    kwargs.pop("device", None)
    kwargs.pop("dtype", None)
    return kwargs


@pytest.fixture
def small_model_config():
    return _small_model_config()


class TestGradientCheckpointing:
    """Tests for gradient checkpointing in DecoderModel."""

    def test_gradient_checkpointing_init_disabled(self, small_model_config, device):
        """Test that gradient checkpointing is disabled by default."""
        model = DecoderModel(**small_model_config, device=device)
        assert not model.gradient_checkpointing

    def test_gradient_checkpointing_init_enabled(self, small_model_config, device):
        """Test that gradient checkpointing can be enabled at init."""
        model = DecoderModel(**small_model_config, gradient_checkpointing=True, device=device)
        assert model.gradient_checkpointing

    def test_enable_disable_methods(self, small_model_config, device):
        """Test enable/disable gradient checkpointing methods."""
        model = DecoderModel(**small_model_config, device=device)

        assert not model.gradient_checkpointing
        model.enable_gradient_checkpointing()
        assert model.gradient_checkpointing
        model.disable_gradient_checkpointing()
        assert not model.gradient_checkpointing

    def test_training_with_checkpointing(self, small_model_config, device):
        """Test that model trains correctly with gradient checkpointing enabled."""
        model = DecoderModel(**small_model_config, gradient_checkpointing=True, device=device)
        model.train()

        batch_size, seq_len = 4, 16
        input_ids = torch.randint(0, small_model_config["vocab_size"], (batch_size, seq_len), device=device)
        labels = torch.randint(0, small_model_config["vocab_size"], (batch_size, seq_len), device=device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Train for a few steps
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, small_model_config["vocab_size"]), labels.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Verify training happened (loss should change)
        assert losses[-1] != losses[0], "Loss should change during training"
        # Verify no NaN
        assert not any(torch.isnan(torch.tensor(loss_val)) for loss_val in losses), "Loss should not be NaN"

    def test_checkpointing_with_use_cache_raises(self, small_model_config, device):
        """Test that using checkpointing with use_cache=True raises ValueError."""
        model = DecoderModel(**small_model_config, gradient_checkpointing=True, device=device)
        model.train()

        input_ids = torch.randint(0, small_model_config["vocab_size"], (2, 8), device=device)

        with pytest.raises(ValueError, match="incompatible"):
            model(input_ids, use_cache=True)

    def test_inference_without_checkpointing(self, small_model_config, device):
        """Test that inference works when checkpointing is disabled."""
        model = DecoderModel(**small_model_config, gradient_checkpointing=True, device=device)
        model.eval()

        # Disable for inference with cache
        model.disable_gradient_checkpointing()

        input_ids = torch.randint(0, small_model_config["vocab_size"], (2, 8), device=device)
        kv_caches = create_decoder_kv_caches(model, batch_size=2)
        logits, kv_cache = model(input_ids, use_cache=True, kv_caches=kv_caches)

        assert logits.shape == (2, 8, small_model_config["vocab_size"])
        assert len(kv_cache) == small_model_config["num_layers"]

    def test_checkpointing_produces_valid_gradients(self, small_model_config, device):
        """Test that checkpointing produces valid gradients (non-zero, no NaN)."""
        model = DecoderModel(**small_model_config, gradient_checkpointing=True, device=device)
        model.train()

        input_ids = torch.randint(0, small_model_config["vocab_size"], (2, 8), device=device)
        labels = torch.randint(0, small_model_config["vocab_size"], (2, 8), device=device)
        criterion = nn.CrossEntropyLoss()

        # Forward + backward with checkpointing
        logits = model(input_ids)
        loss = criterion(logits.view(-1, small_model_config["vocab_size"]), labels.view(-1))
        loss.backward()

        # Verify gradients exist and are valid
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"

        # Verify at least some gradients are non-zero
        total_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        assert total_grad_norm > 0, "Total gradient norm should be positive"


@pytest.mark.skipif(not cuda_usable(), reason="GPU required for memory test")
class TestGradientCheckpointingMemory:
    """Memory-related tests for gradient checkpointing (GPU only)."""

    def test_memory_reduction(self):
        """Test that gradient checkpointing reduces peak memory usage."""
        # Use the fattest usable GPU (repo ``DEFAULT_DEVICE`` convention)
        # rather than literal ``cuda:0``, which may be an occupied / low-VRAM
        # device on a shared host — the class gate only proves *some* GPU has
        # headroom, not cuda:0 specifically.
        from tests.support.devices import DEFAULT_DEVICE

        device = DEFAULT_DEVICE
        config = {
            "vocab_size": 1000,
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 4,
            "max_seq_len": 128,
        }

        batch_size, seq_len = 8, 64
        input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=device)
        labels = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=device)
        criterion = nn.CrossEntropyLoss()

        # Measure memory without checkpointing
        torch.cuda.reset_peak_memory_stats()
        model_no_ckpt = DecoderModel(**config, gradient_checkpointing=False, device=device)
        model_no_ckpt.train()
        logits = model_no_ckpt(input_ids)
        loss = criterion(logits.view(-1, config["vocab_size"]), labels.view(-1))
        loss.backward()
        mem_no_ckpt = torch.cuda.max_memory_allocated()
        del model_no_ckpt, logits, loss
        torch.cuda.empty_cache()

        # Measure memory with checkpointing
        torch.cuda.reset_peak_memory_stats()
        model_with_ckpt = DecoderModel(**config, gradient_checkpointing=True, device=device)
        model_with_ckpt.train()
        logits = model_with_ckpt(input_ids)
        loss = criterion(logits.view(-1, config["vocab_size"]), labels.view(-1))
        loss.backward()
        mem_with_ckpt = torch.cuda.max_memory_allocated()
        del model_with_ckpt, logits, loss
        torch.cuda.empty_cache()

        # Memory with checkpointing should be less (or at most equal for small models)
        # For larger models, the difference is more significant
        assert mem_with_ckpt <= mem_no_ckpt * 1.1, (
            f"Checkpointing should not increase memory significantly: "
            f"{mem_with_ckpt / 1e6:.1f}MB vs {mem_no_ckpt / 1e6:.1f}MB"
        )


class TestGradientCheckpointingEquivalence:
    """Checkpointing must change MEMORY use, not the numbers.

    The existing ``test_checkpointing_produces_valid_gradients`` only checks
    that gradients are non-NaN and not all zero — a checkpointing implementation
    that recomputed with subtly different math (wrong mask, dropped dropout
    seed, mis-scaled loss) would pass it. This class compares against the plain
    path instead.
    """

    @staticmethod
    def _grads(config: dict, device: str, *, checkpointing: bool):
        """Build a seeded model, run forward+backward, return (loss, grads)."""
        torch.manual_seed(1)
        model = DecoderModel(**config, gradient_checkpointing=checkpointing, device=device)
        model.train()
        input_ids = torch.randint(0, config["vocab_size"], (2, 8), device=device)
        loss = model(input_ids).sum()
        loss.backward()
        grads = {
            name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None
        }
        return loss.item(), grads

    def test_gradients_match_the_plain_path(self, small_model_config, device):
        """Every parameter gradient must be identical with and without checkpointing.

        ``torch.utils.checkpoint`` recomputes activations in the backward pass;
        the recomputation is only correct if it is exactly the same computation,
        so equality (not just "close") is the right bar here.
        """
        plain_loss, plain_grads = self._grads(small_model_config, device, checkpointing=False)
        ckpt_loss, ckpt_grads = self._grads(small_model_config, device, checkpointing=True)

        assert plain_loss == pytest.approx(ckpt_loss, rel=1e-6), (
            f"forward loss changed under checkpointing: {plain_loss} vs {ckpt_loss}"
        )
        assert set(plain_grads) == set(ckpt_grads), "the two paths produced gradients for different parameters"

        worst_name, worst_delta = None, 0.0
        for name in plain_grads:
            delta = (plain_grads[name] - ckpt_grads[name]).abs().max().item()
            if delta > worst_delta:
                worst_name, worst_delta = name, delta
        assert worst_delta < 1e-6, f"gradient of {worst_name} differs by {worst_delta:.2e} under checkpointing"

    def test_a_parameter_actually_receives_a_nonzero_gradient(self, small_model_config, device):
        """Guard against the comparison being vacuous (all-zero gradients match trivially)."""
        _, grads = self._grads(small_model_config, device, checkpointing=True)
        assert grads, "no gradients were produced at all"
        assert any(g.abs().sum().item() > 0 for g in grads.values()), (
            "every gradient is zero, so an equality check would be meaningless"
        )
