# tests/core/test_paged_kv_cache.py
import pytest
import torch
from llm.core.paged_attention.paged_kv_cache import PagedKVCache


# Cache device to avoid repeated CUDA checks (session-level)
_cached_device = None


def get_device():
    """Get available device: cuda if available, else cpu. Cached for performance."""
    global _cached_device
    if _cached_device is None:
        _cached_device = "cuda" if torch.cuda.is_available() else "cpu"
    return _cached_device


# Use a fixture to get device for all tests
@pytest.fixture
def device():
    return get_device()


def test_paged_kv_cache_init():
    """Test PagedKVCache initialization with various configurations."""
    dev = get_device()
    cache = PagedKVCache(
        num_layers=2,
        num_kv_heads=4,
        head_dim=16,
        num_blocks=32,
        block_size=16,
        device=dev,
        dtype=torch.float16,
    )
    assert cache.k_cache.shape == (2, 32, 4, 16, 16)
    assert cache.v_cache.shape == (2, 32, 4, 16, 16)
    assert cache.num_layers == 2
    assert cache.num_kv_heads == 4
    assert cache.block_size == 16


def test_paged_kv_cache_init_different_params():
    """Test initialization with different parameter combinations."""
    # Small config
    cache = PagedKVCache(num_layers=1, num_kv_heads=1, head_dim=8, num_blocks=4, block_size=4, device=get_device())
    assert cache.k_cache.shape == (1, 4, 1, 4, 8)

    # Large config
    cache = PagedKVCache(
        num_layers=12, num_kv_heads=8, head_dim=128, num_blocks=512, block_size=32, device=get_device()
    )
    assert cache.k_cache.shape == (12, 512, 8, 32, 128)


def test_update_allocates_single_block():
    """Test that update allocates correct number of blocks for small sequences."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=get_device(),
    )
    # 4 tokens = 1 block (block_size=4)
    k = torch.randn(1, 4, 2, 8)
    v = torch.randn(1, 4, 2, 8)

    block_ids = cache.update(seq_id=1, k_new=k, v_new=v)

    assert len(block_ids) == 1
    assert block_ids == [0]


def test_update_allocates_multiple_blocks():
    """Test block allocation for sequences spanning multiple blocks."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=get_device(),
    )
    # 10 tokens = 3 blocks (4 + 4 + 2)
    k = torch.randn(1, 10, 2, 8)
    v = torch.randn(1, 10, 2, 8)

    block_ids = cache.update(seq_id=1, k_new=k, v_new=v)

    assert len(block_ids) == 3


def test_update_exact_block_boundary():
    """Test sequence that exactly fills whole blocks."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=get_device(),
    )
    # Exactly 8 tokens = 2 blocks
    k = torch.randn(1, 8, 2, 8)
    v = torch.randn(1, 8, 2, 8)

    block_ids = cache.update(seq_id=1, k_new=k, v_new=v)

    assert len(block_ids) == 2


def test_multiple_sequences():
    """Test allocating KV cache for multiple sequences."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=get_device(),
    )

    # First sequence: 4 tokens = 1 block
    k1 = torch.randn(1, 4, 2, 8)
    v1 = torch.randn(1, 4, 2, 8)
    block_ids_1 = cache.update(seq_id=1, k_new=k1, v_new=v1)

    # Second sequence: 8 tokens = 2 blocks
    k2 = torch.randn(1, 8, 2, 8)
    v2 = torch.randn(1, 8, 2, 8)
    block_ids_2 = cache.update(seq_id=2, k_new=k2, v_new=v2)

    assert block_ids_1 == [0]
    assert block_ids_2 == [1, 2]


def test_free_sequence():
    """Test freeing a sequence returns blocks to pool."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=get_device(),
    )
    k = torch.randn(1, 4, 2, 8)
    v = torch.randn(1, 4, 2, 8)

    cache.update(seq_id=1, k_new=k, v_new=v)
    assert cache.block_manager.num_free_blocks == 7

    cache.free(seq_id=1)
    assert cache.block_manager.num_free_blocks == 8


def test_free_sequence_updates_block_table():
    """Test that freeing clears block table."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=get_device(),
    )
    k = torch.randn(1, 8, 2, 8)  # 2 blocks
    v = torch.randn(1, 8, 2, 8)

    cache.update(seq_id=1, k_new=k, v_new=v)
    assert len(cache.get_block_table(1)) == 2

    cache.free(seq_id=1)
    # After free, sequence no longer exists, so raises error
    with pytest.raises(ValueError, match="does not exist"):
        cache.get_block_table(1)


def test_oom_when_no_blocks():
    """Test that update raises OOM when no blocks available."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=2,
        block_size=4,
        device=get_device(),
    )
    # Use up all blocks: 4 tokens = 1 block, 2 blocks total
    k1 = torch.randn(1, 4, 2, 8)
    v1 = torch.randn(1, 4, 2, 8)
    cache.update(seq_id=1, k_new=k1, v_new=v1)

    k2 = torch.randn(1, 4, 2, 8)
    v2 = torch.randn(1, 4, 2, 8)
    cache.update(seq_id=2, k_new=k2, v_new=v2)

    # Now no free blocks left
    assert cache.block_manager.num_free_blocks == 0

    # This should fail - no blocks left
    k3 = torch.randn(1, 4, 2, 8)
    v3 = torch.randn(1, 4, 2, 8)

    with pytest.raises(RuntimeError, match="No free blocks"):
        cache.update(seq_id=3, k_new=k3, v_new=v3)


def test_get_block_table():
    """Test retrieving block table for a sequence."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=get_device(),
    )
    k = torch.randn(1, 8, 2, 8)
    v = torch.randn(1, 8, 2, 8)
    block_ids = cache.update(seq_id=5, k_new=k, v_new=v)

    assert cache.get_block_table(5) == block_ids


def test_get_block_table_unknown_seq():
    """Test get_block_table for non-existent sequence."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=get_device(),
    )
    # BlockManager raises error for unknown sequence - this is expected behavior
    with pytest.raises(ValueError, match="does not exist"):
        cache.get_block_table(999)


def test_get_kv_slice():
    """Test retrieving a slice of KV cache."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=get_device(),
    )

    # Insert 8 tokens = 2 blocks
    k = torch.randn(1, 8, 2, 8)
    v = torch.randn(1, 8, 2, 8)
    cache.update(seq_id=1, k_new=k, v_new=v)

    # Get first 4 tokens
    k_slice, v_slice = cache.get(seq_id=1, start_idx=0, end_idx=4)
    assert k_slice.shape == (2, 4, 8)
    assert v_slice.shape == (2, 4, 8)


def test_get_kv_slice_partial_block():
    """Test retrieving tokens that span block boundaries."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=get_device(),
    )

    # Insert 8 tokens = 2 blocks
    k = torch.randn(1, 8, 2, 8)
    v = torch.randn(1, 8, 2, 8)
    cache.update(seq_id=1, k_new=k, v_new=v)

    # Get tokens 2-6 (spans block 0 and block 1)
    k_slice, v_slice = cache.get(seq_id=1, start_idx=2, end_idx=6)
    assert k_slice.shape == (2, 4, 8)


def test_free_nonexistent_sequence():
    """Test freeing a sequence that doesn't exist."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=get_device(),
    )
    # Should not raise error
    cache.free(seq_id=999)


def test_sequential_allocate_free():
    """Test repeated allocate-free cycles."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=4,
        block_size=4,
        device=get_device(),
    )

    for i in range(4):
        k = torch.randn(1, 4, 2, 8)
        v = torch.randn(1, 4, 2, 8)
        cache.update(seq_id=i, k_new=k, v_new=v)

    assert cache.block_manager.num_free_blocks == 0

    # Free all
    for i in range(4):
        cache.free(seq_id=i)

    assert cache.block_manager.num_free_blocks == 4


def test_block_manager_integration():
    """Test that PagedKVCache correctly uses BlockManager."""
    cache = PagedKVCache(
        num_layers=2,  # 2 layers
        num_kv_heads=2,
        head_dim=8,
        num_blocks=4,
        block_size=4,
        device=get_device(),
    )

    # Verify we have 2 allocators (one per layer)
    assert len(cache.block_manager.allocators) == 2

    k = torch.randn(1, 4, 2, 8)
    v = torch.randn(1, 4, 2, 8)
    cache.update(seq_id=1, k_new=k, v_new=v)

    # Both layers should have allocated
    for alloc in cache.block_manager.allocators:
        assert alloc.num_free_blocks == 3


def test_large_sequence_single_block():
    """Test a large sequence that uses many blocks."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=4,
        head_dim=64,
        num_blocks=64,
        block_size=16,
        device=get_device(),
    )

    # 64 tokens = 4 blocks
    k = torch.randn(1, 64, 4, 64)
    v = torch.randn(1, 64, 4, 64)
    block_ids = cache.update(seq_id=1, k_new=k, v_new=v)

    assert len(block_ids) == 4


def test_cache_dtype_preserved():
    """Test that cache dtype is preserved correctly."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=4,
        block_size=4,
        device=get_device(),
        dtype=torch.float32,
    )
    assert cache.k_cache.dtype == torch.float32
    assert cache.v_cache.dtype == torch.float32

    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=4,
        block_size=4,
        device=get_device(),
        dtype=torch.bfloat16,
    )
    assert cache.k_cache.dtype == torch.bfloat16
