import torch

from llm.core.paged_attention.paged_kv_cache import PagedKVCache


def test_paged_kv_cache_init():
    cache = PagedKVCache(
        num_layers=2,
        num_kv_heads=4,
        head_dim=16,
        num_blocks=32,
        block_size=16,
        device="cpu",
        dtype=torch.float16,
    )
    assert cache.k_cache.shape == (2, 32, 4, 16, 16)


def test_update_allocates_blocks():
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device="cpu",
    )
    # [batch, tokens, heads, head_dim]
    k = torch.randn(1, 4, 2, 8)
    v = torch.randn(1, 4, 2, 8)

    block_ids = cache.update(seq_id=1, k_new=k, v_new=v)

    assert len(block_ids) == 1  # 4 tokens = 1 block
    assert block_ids == [0]


def test_free_sequence():
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device="cpu",
    )
    k = torch.randn(1, 4, 2, 8)
    v = torch.randn(1, 4, 2, 8)

    cache.update(seq_id=1, k_new=k, v_new=v)
    assert cache.block_manager.num_free_blocks == 7

    cache.free(seq_id=1)
    assert cache.block_manager.num_free_blocks == 8
