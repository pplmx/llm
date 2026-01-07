"""
Tests for Sliding Window Attention.
"""

import pytest
import torch

from llm.core.attn.mha import MultiHeadAttention
from llm.models.decoder import DecoderModel


class TestMHAWithWindowSize:
    """Tests for MultiHeadAttention with sliding window."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_mha_with_window_size(self, device):
        """Test MHA with window_size parameter."""
        mha = MultiHeadAttention(
            hidden_size=64,
            num_heads=4,
            window_size=4,
            device=device,
        )

        x = torch.randn(2, 16, 64, device=device)
        output = mha(x)

        assert output.shape == x.shape

    def test_mha_window_size_none_default(self, device):
        """Test that MHA defaults to no window size."""
        mha = MultiHeadAttention(hidden_size=64, num_heads=4, device=device)
        assert mha.window_size is None

    def test_mha_window_reduces_memory(self, device):
        """Test that window attention works (not testing actual memory reduction here)."""
        mha_full = MultiHeadAttention(hidden_size=64, num_heads=4, device=device)
        mha_window = MultiHeadAttention(hidden_size=64, num_heads=4, window_size=4, device=device)

        x = torch.randn(2, 32, 64, device=device)

        # Both should produce valid outputs
        out_full = mha_full(x)
        out_window = mha_window(x)

        assert out_full.shape == out_window.shape
        # Outputs should differ
        assert not torch.allclose(out_full, out_window)


class TestDecoderWithWindowSize:
    """Tests for DecoderModel with sliding window attention."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_decoder_model_with_window_size(self, device):
        """Test DecoderModel forward pass with window_size."""
        model = DecoderModel(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            max_seq_len=32,
            window_size=8,
        ).to(device)

        input_ids = torch.randint(0, 100, (2, 16), device=device)
        output = model(input_ids)

        assert output.shape == (2, 16, 100)

    def test_decoder_training_with_window_size(self, device):
        """Test that DecoderModel trains with window_size."""
        model = DecoderModel(
            vocab_size=50,
            hidden_size=32,
            num_layers=1,
            num_heads=2,
            max_seq_len=16,
            window_size=4,
        ).to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        input_ids = torch.randint(0, 50, (2, 8), device=device)
        labels = torch.randint(0, 50, (2, 8), device=device)

        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, 50), labels.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should change (training happening)
        assert losses[0] != losses[-1] or len(set(losses)) > 1
