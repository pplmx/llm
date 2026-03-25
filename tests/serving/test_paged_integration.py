import pytest
import torch

from llm.core.paged_attention.attention import paged_attention_forward
from llm.core.paged_attention.paged_kv_cache import PagedKVCache


def get_device():
    """Get available device: cuda if available, else cpu."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_end_to_end_paged_inference():
    """Test paged attention workflow with multiple sequences."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=16,
        num_blocks=32,
        block_size=16,
        device=get_device(),
        dtype=torch.float32,
    )

    seq1_k = torch.randn(1, 10, 2, 16)
    seq1_v = torch.randn(1, 10, 2, 16)
    block_ids_1 = cache.update(seq_id=1, k_new=seq1_k, v_new=seq1_v)

    seq2_k = torch.randn(1, 8, 2, 16)
    seq2_v = torch.randn(1, 8, 2, 16)
    block_ids_2 = cache.update(seq_id=2, k_new=seq2_k, v_new=seq2_v)

    max_blocks = max(len(block_ids_1), len(block_ids_2))
    block_tables = torch.zeros(2, max_blocks, dtype=torch.long)
    block_tables[0, : len(block_ids_1)] = torch.tensor(block_ids_1)
    block_tables[1, : len(block_ids_2)] = torch.tensor(block_ids_2)
    seq_lens = torch.tensor([10, 8])

    q = torch.randn(2, 4, 1, 16)
    output = paged_attention_forward(
        q=q,
        k_cache=cache.k_cache,
        v_cache=cache.v_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        num_kv_heads=2,
    )

    assert output.shape == (2, 4, 1, 16)

    cache.free(seq_id=1)
    cache.free(seq_id=2)


@pytest.mark.quick
def test_prefix_cache_integration_flow():
    """Test full prefix cache integration flow."""
    from llm.core.paged_attention.paged_kv_cache import PagedKVCache

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=16,
        num_blocks=64,
        block_size=16,
        device=dev,
        enable_prefix_cache=True,
        max_prefixes=10,
    )

    prefix_tokens = [1, 2, 3, 4, 5]
    k_prefix = torch.randn(1, 5, 2, 16)
    v_prefix = torch.randn(1, 5, 2, 16)

    block_ids = cache.update(seq_id=1, k_new=k_prefix, v_new=v_prefix)
    cache.add_prefix(seq_id=1, prefix_tokens=prefix_tokens, block_ids=block_ids)

    cached_blocks = cache.try_get_prefix_blocks(prefix_tokens)
    assert cached_blocks == block_ids

    k_new = torch.randn(1, 3, 2, 16)
    v_new = torch.randn(1, 3, 2, 16)
    new_block_ids = cache.update(seq_id=2, k_new=k_new, v_new=v_new)

    assert cache.get_block_table(1) == block_ids
    assert cache.get_block_table(2) == new_block_ids
