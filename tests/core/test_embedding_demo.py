"""
Embedding Layer Demo Tests

Tests EmbeddingLayer with both sinusoidal and learned positional encodings.
"""

import torch

from llm.core.embedding import EmbeddingLayer


def test_embedding_layer_sinusoidal():
    """Test EmbeddingLayer with sinusoidal positional encoding."""
    vocab_size = 1000
    hidden_size = 512
    max_seq_len = 100
    batch_size = 4
    seq_len = 50

    device = torch.device("cpu")
    dtype = torch.float32

    embedding_layer = EmbeddingLayer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_seq_len=max_seq_len,
        pos_encoding_learned=False,
        dropout_p=0.1,
        padding_idx=0,
        device=device,
        dtype=dtype,
    )

    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    dummy_input_ids[0, 0] = 0  # Test padding

    output = embedding_layer(dummy_input_ids)

    assert output.shape == (batch_size, seq_len, hidden_size)
    assert output.dtype == dtype
    assert str(output.device) == str(device)


def test_embedding_layer_learned():
    """Test EmbeddingLayer with learned positional encoding."""
    vocab_size = 1000
    hidden_size = 512
    max_seq_len = 100
    batch_size = 4
    seq_len = 50

    device = torch.device("cpu")
    dtype = torch.float32

    embedding_layer = EmbeddingLayer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_seq_len=max_seq_len,
        pos_encoding_learned=True,
        dropout_p=0.1,
        padding_idx=0,
        device=device,
        dtype=dtype,
    )

    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    output = embedding_layer(dummy_input_ids)

    assert output.shape == (batch_size, seq_len, hidden_size)
    assert output.dtype == dtype
    assert str(output.device) == str(device)

    # Check that positional embedding exists
    assert hasattr(embedding_layer.positional_encoding, "pos_embedding")
