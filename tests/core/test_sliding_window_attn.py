"""
Tests for Sliding Window Attention.
"""

import torch

from llm.core.attn.mha import MultiHeadAttention
from llm.models.decoder import DecoderModel


class TestMHAWithWindowSize:
    """Tests for MultiHeadAttention with sliding window."""

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


class TestSDPAWindowingDecode:
    """Sliding-window mask must use the query block's ABSOLUTE position.

    During KV-cache decode the query is a single token at absolute position
    ``seq_len_k - 1`` while its row index within the block is 0. The window
    mask previously compared the *relative* query row against *absolute* key
    columns, so a decode step attended the OLDEST ``window_size`` keys
    instead of the ``window_size`` keys before the current position.
    """

    def test_decode_attends_recent_window_not_oldest(self, device):
        from llm.core.attn.sdpa import sdpa

        # 20 cached keys, one decode query at absolute position 19, window=4.
        q = torch.zeros(1, 1, 1, 8, device=device)  # uniform attention
        k = torch.ones(1, 1, 20, 8, device=device)
        v = torch.arange(20, dtype=torch.float32, device=device).reshape(1, 1, 20, 1).expand(-1, -1, -1, 8).clone()

        out = sdpa(q, k, v, is_causal=False, window_size=4)

        # Mean of the attended keys: correct = [15..19] -> 17; the old
        # absolute-vs-relative bug attended [0..4] -> 2.
        assert abs(out[0, 0, 0, 0].item() - 17.0) < 1e-5

    def test_prefill_window_still_keeps_recent(self, device):
        """Pure prefill (Sq == Sk, query block at positions 0..Sq-1) keeps the
        recent window — unchanged by the decode fix."""
        from llm.core.attn.sdpa import sdpa

        max_len = 8
        q = torch.zeros(1, 1, max_len, 8, device=device)
        k = torch.ones(1, 1, max_len, 8, device=device)
        v = (
            torch.arange(max_len, dtype=torch.float32, device=device)
            .reshape(1, 1, max_len, 1)
            .expand(-1, -1, -1, 8)
            .clone()
        )

        out = sdpa(q, k, v, is_causal=False, window_size=2)

        # Last query row (position 7) attends keys [5..7] -> mean 6.
        assert abs(out[0, 0, 7, 0].item() - 6.0) < 1e-5
        # First query row (position 0) attends keys [0..2] -> mean 1.
        assert abs(out[0, 0, 0, 0].item() - 1.0) < 1e-5
