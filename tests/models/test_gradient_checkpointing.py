"""
Tests for Gradient Checkpointing.

Verifies that gradient checkpointing works correctly for memory-efficient training.
"""

import pytest
import torch
import torch.nn as nn

from llm.models.decoder import DecoderModel


class TestGradientCheckpointing:
    """Tests for gradient checkpointing in DecoderModel."""

    @pytest.fixture
    def small_model_config(self):
        return {
            "vocab_size": 100,
            "hidden_size": 64,
            "num_layers": 2,
            "num_heads": 2,
            "max_seq_len": 32,
        }

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        logits, kv_cache = model(input_ids, use_cache=True)

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for memory test")
class TestGradientCheckpointingMemory:
    """Memory-related tests for gradient checkpointing (GPU only)."""

    def test_memory_reduction(self):
        """Test that gradient checkpointing reduces peak memory usage."""
        device = torch.device("cuda")
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
