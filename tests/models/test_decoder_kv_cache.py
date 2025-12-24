import pytest
import torch

from llm.models.decoder import DecoderModel


@pytest.mark.slow
def test_kv_cache_consistency():
    """
    Verify that incremental generation with KV cache produces the same hidden states
    and logits as a full-sequence forward pass.
    """
    vocab_size = 100
    hidden_size = 64
    num_layers = 2
    num_heads = 4
    seq_len = 5

    model = DecoderModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        is_causal=True,
        pos_encoding_learned=True,
        mlp_dropout_p=0.0,
        attn_dropout_p=0.0,
        embedding_dropout_p=0.0,
    )
    model.eval()

    input_ids = torch.randint(0, vocab_size, (1, seq_len))

    # 1. Full-sequence forward pass
    with torch.no_grad():
        full_logits = model(input_ids)

    # 2. Incremental forward pass with KV cache
    past_key_values = None
    incremental_logits = []

    for i in range(seq_len):
        curr_input_id = input_ids[:, i : i + 1]
        with torch.no_grad():
            logits, past_key_values = model(curr_input_id, past_key_values=past_key_values, use_cache=True)
            incremental_logits.append(logits)

    incremental_logits = torch.cat(incremental_logits, dim=1)

    # Compare
    assert torch.allclose(full_logits, incremental_logits, atol=1e-5), (
        "KV cache consistency failed: incremental logits do not match full-sequence logits."
    )


@pytest.mark.slow
def test_kv_cache_dimensions():
    """
    Verify the dimensions of the returned KV cache.
    """
    vocab_size = 50
    hidden_size = 32
    num_layers = 2
    num_heads = 4
    num_kv_heads = 2
    seq_len = 3

    model = DecoderModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        is_causal=True,
    )
    model.eval()

    input_ids = torch.randint(0, vocab_size, (1, seq_len))

    with torch.no_grad():
        _, kv_cache = model(input_ids, use_cache=True)

    # kv_cache is a list of tuples (key, value) for each layer
    assert len(kv_cache) == num_layers
    for layer_cache in kv_cache:
        k, v = layer_cache
        # Expected shape: [batch, num_kv_heads, seq_len, head_dim]
        # head_dim = hidden_size // num_heads = 32 // 4 = 8
        assert k.shape == (1, num_kv_heads, seq_len, 8)
        assert v.shape == (1, num_kv_heads, seq_len, 8)
