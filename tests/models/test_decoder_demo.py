"""
Decoder Model Demo Tests

Tests DecoderModel with Pre-LN and Post-LN configurations.
"""

import torch

from llm.models.decoder import DecoderModel


def test_decoder_pre_ln():
    """Test DecoderModel with Pre-LN configuration."""
    device = torch.device("cpu")
    dtype = torch.float32

    vocab_size = 100
    hidden_size = 64
    num_layers = 2
    num_heads = 4
    max_seq_len = 128
    batch_size = 4
    seq_len = 50

    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    decoder = DecoderModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        norm_first=True,
        device=device,
        dtype=dtype,
    )
    decoder.eval()

    output = decoder(dummy_input_ids)

    assert output.shape == (batch_size, seq_len, vocab_size)
    assert decoder.final_norm is not None


def test_decoder_pre_ln_with_mask():
    """Test Pre-LN DecoderModel with attention mask."""
    device = torch.device("cpu")
    dtype = torch.float32

    vocab_size = 100
    hidden_size = 64
    num_layers = 2
    num_heads = 4
    max_seq_len = 128
    batch_size = 4
    seq_len = 50

    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    # Create padding mask
    padding_mask = torch.zeros(batch_size, 1, 1, seq_len, device=device, dtype=torch.bool)
    padding_mask[0, 0, 0, -10:] = True

    decoder = DecoderModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        norm_first=True,
        device=device,
        dtype=dtype,
    )
    decoder.eval()

    output = decoder(dummy_input_ids, attn_mask=padding_mask)

    assert output.shape == (batch_size, seq_len, vocab_size)


def test_decoder_post_ln():
    """Test DecoderModel with Post-LN configuration."""
    device = torch.device("cpu")
    dtype = torch.float32

    vocab_size = 100
    hidden_size = 64
    num_layers = 2
    num_heads = 4
    max_seq_len = 128
    batch_size = 4
    seq_len = 50

    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    decoder = DecoderModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        norm_first=False,
        device=device,
        dtype=dtype,
    )
    decoder.eval()

    output = decoder(dummy_input_ids)

    assert output.shape == (batch_size, seq_len, vocab_size)
    assert decoder.final_norm is None


def test_decoder_post_ln_with_mask():
    """Test Post-LN DecoderModel with attention mask."""
    device = torch.device("cpu")
    dtype = torch.float32

    vocab_size = 100
    hidden_size = 64
    num_layers = 2
    num_heads = 4
    max_seq_len = 128
    batch_size = 4
    seq_len = 50

    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    # Create padding mask
    padding_mask = torch.zeros(batch_size, 1, 1, seq_len, device=device, dtype=torch.bool)
    padding_mask[0, 0, 0, -10:] = True

    decoder = DecoderModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        norm_first=False,
        device=device,
        dtype=dtype,
    )
    decoder.eval()

    output = decoder(dummy_input_ids, attn_mask=padding_mask)

    assert output.shape == (batch_size, seq_len, vocab_size)
