import pytest
import torch

from llm.core.kv_cache import KVCache, create_decoder_kv_caches
from llm.models.decoder import DecoderModel
from tests.support.models import decoder_model_kwargs


@pytest.mark.slow
def test_kv_cache_consistency():
    """Incremental cached forward must match full-sequence forward."""
    kwargs = decoder_model_kwargs(
        vocab_size=100,
        num_layers=2,
        is_causal=True,
        pos_encoding_learned=True,
    )
    seq_len = 5
    model = DecoderModel(**kwargs)
    model.eval()

    input_ids = torch.randint(0, kwargs["vocab_size"], (1, seq_len), device=next(model.parameters()).device)

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

    assert torch.allclose(full_logits, incremental_logits, atol=1e-5)


@pytest.mark.slow
def test_kv_cache_dimensions():
    """Verify KVCache tensor shapes after a cached forward pass."""
    kwargs = decoder_model_kwargs(
        vocab_size=50,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        is_causal=True,
    )
    seq_len = 3
    model = DecoderModel(**kwargs)
    model.eval()

    input_ids = torch.randint(0, kwargs["vocab_size"], (1, seq_len), device=next(model.parameters()).device)
    kv_caches = create_decoder_kv_caches(model, batch_size=1)

    with torch.no_grad():
        _, kv_caches = model(input_ids, kv_caches=kv_caches, use_cache=True)

    assert len(kv_caches) == kwargs["num_layers"]
    head_dim = kwargs["hidden_size"] // kwargs["num_heads"]
    for cache in kv_caches:
        assert type(cache) is KVCache
        assert cache.seq_len == seq_len
        assert cache.k_cache.shape[1] == kwargs["num_kv_heads"]
        assert cache.k_cache.shape[3] == head_dim
