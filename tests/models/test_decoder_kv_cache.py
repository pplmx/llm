import pytest
import torch

from llm.core.kv_cache import KVCache, create_decoder_kv_caches
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

    with torch.no_grad():
        full_logits = model(input_ids)

    kv_caches = create_decoder_kv_caches(model, batch_size=1)
    incremental_logits = []

    for i in range(seq_len):
        curr_input_id = input_ids[:, i : i + 1]
        with torch.no_grad():
            logits, kv_caches = model(curr_input_id, kv_caches=kv_caches, use_cache=True)
            incremental_logits.append(logits)

    incremental_logits = torch.cat(incremental_logits, dim=1)

    assert torch.allclose(full_logits, incremental_logits, atol=1e-5), (
        "KV cache consistency failed: incremental logits do not match full-sequence logits."
    )


@pytest.mark.slow
def test_kv_cache_dimensions():
    """Verify KVCache state after a cached forward pass."""
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
    kv_caches = create_decoder_kv_caches(model, batch_size=1)

    with torch.no_grad():
        _, kv_caches = model(input_ids, kv_caches=kv_caches, use_cache=True)

    assert len(kv_caches) == num_layers
    for cache in kv_caches:
        assert isinstance(cache, KVCache)
        assert cache.seq_len == seq_len
        assert cache.k_cache.shape[1] == num_kv_heads
        assert cache.k_cache.shape[3] == hidden_size // num_heads
