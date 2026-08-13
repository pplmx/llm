# tests/core/test_paged_kv_cache.py
import pytest
import torch

from llm.core.paged_attention.paged_kv_cache import PagedKVCache
from tests.support.devices import DEFAULT_DEVICE, cuda_usable

# Prefer GPU via DEFAULT_DEVICE; fall back to CPU when CUDA is unusable.
DEVICE = DEFAULT_DEVICE
LARGE_INIT_MIN_FREE_BYTES = 2 * 1024**3


def test_paged_kv_cache_init():
    """Test PagedKVCache initialization with various configurations."""
    cache = PagedKVCache(
        num_layers=2,
        num_kv_heads=4,
        head_dim=16,
        num_blocks=32,
        block_size=16,
        device=DEVICE,
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
    cache = PagedKVCache(num_layers=1, num_kv_heads=1, head_dim=8, num_blocks=4, block_size=4, device=DEVICE)
    assert cache.k_cache.shape == (1, 4, 1, 4, 8)

    # Large config (~1.5 GiB for float32 K+V) — skip when the chosen GPU is tight.
    if DEVICE.type == "cuda" and not cuda_usable(DEVICE, min_free_bytes=LARGE_INIT_MIN_FREE_BYTES):
        pytest.skip(f"{DEVICE} needs at least 2 GiB free for large PagedKVCache init")
    cache = PagedKVCache(num_layers=12, num_kv_heads=8, head_dim=128, num_blocks=512, block_size=32, device=DEVICE)
    assert cache.k_cache.shape == (12, 512, 8, 32, 128)


def test_update_allocates_single_block():
    """Test that update allocates correct number of blocks for small sequences."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
    )
    # 4 tokens = 1 block (block_size=4)
    k = torch.randn(1, 4, 2, 8, device=DEVICE)
    v = torch.randn(1, 4, 2, 8, device=DEVICE)

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
        device=DEVICE,
    )
    # 10 tokens = 3 blocks (4 + 4 + 2)
    k = torch.randn(1, 10, 2, 8, device=DEVICE)
    v = torch.randn(1, 10, 2, 8, device=DEVICE)

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
        device=DEVICE,
    )
    # Exactly 8 tokens = 2 blocks
    k = torch.randn(1, 8, 2, 8, device=DEVICE)
    v = torch.randn(1, 8, 2, 8, device=DEVICE)

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
        device=DEVICE,
    )

    # First sequence: 4 tokens = 1 block
    k1 = torch.randn(1, 4, 2, 8, device=DEVICE)
    v1 = torch.randn(1, 4, 2, 8, device=DEVICE)
    block_ids_1 = cache.update(seq_id=1, k_new=k1, v_new=v1)

    # Second sequence: 8 tokens = 2 blocks
    k2 = torch.randn(1, 8, 2, 8, device=DEVICE)
    v2 = torch.randn(1, 8, 2, 8, device=DEVICE)
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
        device=DEVICE,
    )
    k = torch.randn(1, 4, 2, 8, device=DEVICE)
    v = torch.randn(1, 4, 2, 8, device=DEVICE)

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
        device=DEVICE,
    )
    k = torch.randn(1, 8, 2, 8, device=DEVICE)  # 2 blocks
    v = torch.randn(1, 8, 2, 8, device=DEVICE)

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
        device=DEVICE,
    )
    # Use up all blocks: 4 tokens = 1 block, 2 blocks total
    k1 = torch.randn(1, 4, 2, 8, device=DEVICE)
    v1 = torch.randn(1, 4, 2, 8, device=DEVICE)
    cache.update(seq_id=1, k_new=k1, v_new=v1)

    k2 = torch.randn(1, 4, 2, 8, device=DEVICE)
    v2 = torch.randn(1, 4, 2, 8, device=DEVICE)
    cache.update(seq_id=2, k_new=k2, v_new=v2)

    # Now no free blocks left
    assert cache.block_manager.num_free_blocks == 0

    # This should fail - no blocks left
    k3 = torch.randn(1, 4, 2, 8, device=DEVICE)
    v3 = torch.randn(1, 4, 2, 8, device=DEVICE)

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
        device=DEVICE,
    )
    k = torch.randn(1, 8, 2, 8, device=DEVICE)
    v = torch.randn(1, 8, 2, 8, device=DEVICE)
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
        device=DEVICE,
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
        device=DEVICE,
    )

    # Insert 8 tokens = 2 blocks
    k = torch.randn(1, 8, 2, 8, device=DEVICE)
    v = torch.randn(1, 8, 2, 8, device=DEVICE)
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
        device=DEVICE,
    )

    # Insert 8 tokens = 2 blocks
    k = torch.randn(1, 8, 2, 8, device=DEVICE)
    v = torch.randn(1, 8, 2, 8, device=DEVICE)
    cache.update(seq_id=1, k_new=k, v_new=v)

    # Get tokens 2-6 (spans block 0 and block 1)
    k_slice, _v_slice = cache.get(seq_id=1, start_idx=2, end_idx=6)
    assert k_slice.shape == (2, 4, 8)


def test_free_nonexistent_sequence():
    """Test freeing a sequence that doesn't exist."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
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
        device=DEVICE,
    )

    for i in range(4):
        k = torch.randn(1, 4, 2, 8, device=DEVICE)
        v = torch.randn(1, 4, 2, 8, device=DEVICE)
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
        device=DEVICE,
    )

    # Verify we have 2 allocators (one per layer)
    assert len(cache.block_manager.allocators) == 2

    k = torch.randn(1, 4, 2, 8, device=DEVICE)
    v = torch.randn(1, 4, 2, 8, device=DEVICE)
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
        device=DEVICE,
    )

    # 64 tokens = 4 blocks
    k = torch.randn(1, 64, 4, 64, device=DEVICE)
    v = torch.randn(1, 64, 4, 64, device=DEVICE)
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
        device=DEVICE,
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
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    assert cache.k_cache.dtype == torch.bfloat16


def test_prefix_cache_init():
    """Test PrefixCache initialization."""
    from llm.core.paged_attention.paged_kv_cache import PrefixCache

    cache = PrefixCache(max_prefixes=5)
    assert cache.max_prefixes == 5
    assert len(cache.cache) == 0


def test_prefix_cache_add_and_get():
    """Test adding and retrieving cached prefix blocks."""
    from llm.core.paged_attention.paged_kv_cache import PrefixCache

    cache = PrefixCache(max_prefixes=5)
    block_ids = [0, 1, 2]
    cache.add("hash123", block_ids)
    assert cache.get("hash123") == block_ids


def test_prefix_cache_miss():
    """Test cache miss returns None."""
    from llm.core.paged_attention.paged_kv_cache import PrefixCache

    cache = PrefixCache(max_prefixes=5)
    assert cache.get("nonexistent") is None


def test_prefix_cache_lru_eviction():
    """Test LRU eviction when cache is full."""

    from llm.core.paged_attention.paged_kv_cache import PrefixCache

    cache = PrefixCache(max_prefixes=2)
    cache.add("hash1", [0])
    cache.add("hash2", [1])
    cache.add("hash3", [2])

    assert cache.get("hash1") is None
    assert cache.get("hash2") == [1]
    assert cache.get("hash3") == [2]


def test_paged_kv_cache_with_prefix_cache():
    """Test PagedKVCache with prefix cache enabled."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
        enable_prefix_cache=True,
    )
    assert cache.enable_prefix_cache is True
    assert cache.prefix_cache is not None


def test_paged_kv_cache_prefix_cache_disabled():
    """Test PagedKVCache with prefix cache disabled."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
        enable_prefix_cache=False,
    )
    assert cache.enable_prefix_cache is False
    assert cache.prefix_cache is None


def test_add_prefix_and_get_prefix():
    """Test adding and retrieving prefix blocks."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
        enable_prefix_cache=True,
    )

    prefix_tokens = [1, 2, 3, 4]
    block_ids = [0, 1]

    cache.add_prefix(seq_id=1, prefix_tokens=prefix_tokens, block_ids=block_ids)

    result = cache.try_get_prefix_blocks(prefix_tokens)
    assert result == block_ids


def test_prefix_cache_hit():
    """Test cache hit scenario."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
        enable_prefix_cache=True,
    )

    prefix_tokens = [1, 2, 3, 4]
    block_ids = [0, 1]

    cache.add_prefix(seq_id=1, prefix_tokens=prefix_tokens, block_ids=block_ids)

    cached = cache.try_get_prefix_blocks(prefix_tokens)
    assert cached is not None
    assert cached == block_ids


def test_paged_kv_cache_prefix_cache_miss():
    """Test cache miss scenario."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
        enable_prefix_cache=True,
    )

    prefix_tokens = [1, 2, 3, 4]

    cached = cache.try_get_prefix_blocks(prefix_tokens)
    assert cached is None


def test_prefix_cache_disabled_returns_none():
    """Test that disabled prefix cache always returns None."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
        enable_prefix_cache=False,
    )

    cached = cache.try_get_prefix_blocks([1, 2, 3])
    assert cached is None

    cache.add_prefix(seq_id=1, prefix_tokens=[1, 2, 3], block_ids=[0, 1])


def test_free_invalidates_prefix_cache_entry():
    """BUG (RIL ISS-071): ``free(seq_id)`` freed a sequence's blocks but left
    its prefix-cache entry (and ``_seq_to_hash``) pointing at the physical
    block IDs.  Once the allocator recycled those blocks for another
    sequence, a later ``try_get_prefix_blocks`` with the same prefix
    returned dangling block IDs — a use-after-free of the KV cache.

    After ``free`` the prefix hash must no longer resolve, and the
    ``_seq_to_hash`` bookkeeping must be dropped.
    """
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
        enable_prefix_cache=True,
    )

    # Sequence 1 consumes blocks and registers a prefix entry for them.
    k = torch.randn(1, 8, 2, 8, device=DEVICE)  # 2 blocks
    v = torch.randn(1, 8, 2, 8, device=DEVICE)
    block_table = cache.update(seq_id=1, k_new=k, v_new=v)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    cache.add_prefix(seq_id=1, prefix_tokens=tokens, block_ids=block_table)

    # Prefix resolves while the sequence is alive.
    assert cache.try_get_prefix_blocks(tokens) == block_table
    assert 1 in cache._seq_to_hash

    # Freeing the sequence must invalidate the prefix entry + bookkeeping.
    cache.free(seq_id=1)
    assert cache.try_get_prefix_blocks(tokens) is None, (
        "prefix entry still resolves after free() — use-after-free: the "
        "cached block_ids belong to a recycled/can-never-own sequence"
    )
    assert 1 not in cache._seq_to_hash


def test_free_nonexistent_seq_then_reuse_does_not_resolve_stale_prefix():
    """Follow-up on ISS-071: after a sequence frees, the allocator hands its
    blocks to a NEW sequence; the old prefix must not alias the new KV.

    This mirrors the real engine flow (sequence completed → free → a later
    request gets the recycled blocks).  Without invalidation-on-free the
    stale hash would return the new sequence's blocks as a "prefix" — the
    paged attention would then attend over the wrong K/V.
    """
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
        enable_prefix_cache=True,
    )

    k1 = torch.randn(1, 8, 2, 8, device=DEVICE)
    v1 = torch.randn(1, 8, 2, 8, device=DEVICE)
    block_table1 = cache.update(seq_id=1, k_new=k1, v_new=v1)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    cache.add_prefix(seq_id=1, prefix_tokens=tokens, block_ids=block_table1)
    cache.free(seq_id=1)

    # A new sequence is allocated from the (recycled) pool.
    k2 = torch.randn(1, 8, 2, 8, device=DEVICE)
    v2 = torch.randn(1, 8, 2, 8, device=DEVICE)
    block_table2 = cache.update(seq_id=2, k_new=k2, v_new=v2)

    # Old prefix must not alias sequence 2's (recycled) blocks.
    assert cache.try_get_prefix_blocks(tokens) is None, (
        "stale prefix resolved to recycled blocks after free (ISS-071 UAF)"
    )
    # Sanity: seq 2 really did land in the pool — its write succeeded and
    # its table is live (whether or not the allocator reused seq 1's exact
    # block ids, the stale prefix must never point at them).
    assert len(block_table2) == len(block_table1)


def test_get_out_of_bounds_start():
    """Test that get raises ValueError for negative start_idx."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
    )
    k = torch.randn(1, 4, 2, 8, device=DEVICE)
    v = torch.randn(1, 4, 2, 8, device=DEVICE)
    cache.update(seq_id=1, k_new=k, v_new=v)

    with pytest.raises(ValueError, match="out of bounds"):
        cache.get(seq_id=1, start_idx=-1, end_idx=2)


def test_get_out_of_bounds_end():
    """Test that get raises ValueError when end_idx exceeds num tokens."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
    )
    k = torch.randn(1, 4, 2, 8, device=DEVICE)
    v = torch.randn(1, 4, 2, 8, device=DEVICE)
    cache.update(seq_id=1, k_new=k, v_new=v)

    with pytest.raises(ValueError, match="out of bounds"):
        cache.get(seq_id=1, start_idx=0, end_idx=10)


def test_get_start_equals_end():
    """Test that get raises ValueError when start_idx == end_idx."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
    )
    k = torch.randn(1, 4, 2, 8, device=DEVICE)
    v = torch.randn(1, 4, 2, 8, device=DEVICE)
    cache.update(seq_id=1, k_new=k, v_new=v)

    with pytest.raises(ValueError, match="must be less than"):
        cache.get(seq_id=1, start_idx=2, end_idx=2)


def test_multi_layer_update_scopes_each_layer_kv():
    """Each layer's update() must write only its own cache slice.

    Regression: ``update()`` indexed ``self.k_cache[:, block_id, ...]``
    (the leading ``:`` is the *layer* axis), so layer 0's K/V was broadcast
    into every layer's slice and the last layer to write clobbered all the
    others — every layer attended over the same (wrong) K/V for num_layers>1.
    It also advanced the shared block-manager token count once per layer, so
    layer 1 wrote at token 1 instead of replaying the same token at its own
    layer.
    """
    cache = PagedKVCache(
        num_layers=2,
        num_kv_heads=1,
        head_dim=2,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
        dtype=torch.float32,
    )
    k0 = torch.randn(1, 1, 1, 2, device=DEVICE)
    v0 = torch.randn(1, 1, 1, 2, device=DEVICE)
    k1 = torch.randn(1, 1, 1, 2, device=DEVICE)
    v1 = torch.randn(1, 1, 1, 2, device=DEVICE)

    # The decoder calls update() once per layer with that layer's own K/V.
    cache.update(seq_id=0, k_new=k0, v_new=v0, layer_idx=0)
    cache.update(seq_id=0, k_new=k1, v_new=v1, layer_idx=1)

    # Each layer keeps its OWN data at the same logical token (offset 0).
    assert torch.allclose(cache.k_cache[0, 0, 0, 0], k0[0, 0, 0])
    assert torch.allclose(cache.k_cache[1, 0, 0, 0], k1[0, 0, 0])
    assert not torch.allclose(cache.k_cache[0, 0, 0, 0], cache.k_cache[1, 0, 0, 0])
    # The block manager advanced the token count exactly once (not per layer).
    assert cache.block_manager.get_num_tokens(0) == 1

    # Layer-scoped reads return the right per-layer slice.
    k_row0, _ = cache.get(seq_id=0, start_idx=0, end_idx=1, layer_idx=0)
    k_row1, _ = cache.get(seq_id=0, start_idx=0, end_idx=1, layer_idx=1)
    assert torch.allclose(k_row0[0, 0], k0[0, 0, 0])
    assert torch.allclose(k_row1[0, 0], k1[0, 0, 0])


# ---------------------------------------------------------------------------
# Paged prefix replay (RIL TASK-065): shared-block staging + COW-on-write.
# ---------------------------------------------------------------------------


def test_stage_prefix_forks_shared_blocks():
    """``stage_prefix`` must create a new sequence that SHARES the cached
    prefix blocks (refcount bump, no data copy) so the model's prefill of the
    suffix reads the cached K/V without recomputing it."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
        enable_prefix_cache=True,
    )
    k = torch.randn(1, 5, 2, 8, device=DEVICE, dtype=torch.float16)
    v = torch.randn(1, 5, 2, 8, device=DEVICE, dtype=torch.float16)
    table1 = cache.update(seq_id=1, k_new=k, v_new=v)
    cache.add_prefix(seq_id=1, prefix_tokens=[1, 2, 3, 4, 5], block_ids=table1)

    staged = cache.stage_prefix(seq_id=2, prefix_block_ids=table1, num_prefix_tokens=4)
    assert staged == table1
    # Both sequences reference the same physical blocks — refcounted, shared.
    for blk in table1:
        assert cache.block_manager.is_block_shared(blk)
    # seq 2 can read the staged prefix content exactly (shared, not copied).
    k1, v1 = cache.get(seq_id=1, start_idx=0, end_idx=4)
    k2, v2 = cache.get(seq_id=2, start_idx=0, end_idx=4)
    assert torch.equal(k1, k2)
    assert torch.equal(v1, v2)


def test_update_cow_on_shared_boundary_block_preserves_owner():
    """Writing a suffix / decode token into a SHARED block (the prefix cache
    owner holds refcount>1) must copy-on-write: allocate a private block,
    copy the old content, and leave the owner's block untouched."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
        enable_prefix_cache=True,
    )
    k = torch.randn(1, 5, 2, 8, device=DEVICE, dtype=torch.float16)
    v = torch.randn(1, 5, 2, 8, device=DEVICE, dtype=torch.float16)
    table1 = cache.update(seq_id=1, k_new=k, v_new=v)
    owner_k_before = cache.get(seq_id=1, start_idx=0, end_idx=5)[0].clone()
    cache.add_prefix(seq_id=1, prefix_tokens=[1, 2, 3, 4, 5], block_ids=table1)

    # seq 2 stages the first 4 tokens (shared block 1 holds token 4..7), then
    # appends its own token at offset 4 → block 1 is shared → must COW.
    cache.stage_prefix(seq_id=2, prefix_block_ids=table1, num_prefix_tokens=4)
    k_new = torch.randn(1, 1, 2, 8, device=DEVICE, dtype=torch.float16)
    v_new = torch.randn(1, 1, 2, 8, device=DEVICE, dtype=torch.float16)
    table2 = cache.update(seq_id=2, k_new=k_new, v_new=v_new)

    # The boundary block was COW'd (different physical id); the untouched
    # leading block is still shared with the owner.
    assert table2 != table1
    assert table2[0] == table1[0]
    assert table2[1] != table1[1]
    # seq 1's cached K/V is byte-identical after seq 2's write.
    owner_k_after = cache.get(seq_id=1, start_idx=0, end_idx=5)[0]
    assert torch.equal(owner_k_before, owner_k_after)
    # seq 2 sees the old prefix content up to the boundary, then its own token.
    # get() yields [num_kv_heads, num_tokens, head_dim].
    k2, _ = cache.get(seq_id=2, start_idx=0, end_idx=5)
    assert torch.equal(k2[:, 0:4, :], owner_k_before[:, 0:4, :])
    assert torch.equal(k2[:, 4, :], k_new[0, 0, :, :])


def test_prefix_hit_idempotent_rewrite_matches_full_prefill():
    """Full-prefix hit semantics: stage with ``num_prefix_tokens = len - 1``
    then append the last prompt token — the rewrite is idempotent, so the
    staged sequence's KV is identical to a fresh full prefill of the prompt."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
        enable_prefix_cache=True,
    )
    # Full prefill of a 7-token prompt (2 blocks).
    k_full = torch.randn(1, 7, 2, 8, device=DEVICE, dtype=torch.float16)
    v_full = torch.randn(1, 7, 2, 8, device=DEVICE, dtype=torch.float16)
    table_owner = cache.update(seq_id=1, k_new=k_full, v_new=v_full)

    # Replay path: another seq stages the same prefix as 6 tokens, then
    # appends the last prompt token's K/V at position 6 (the boundary block
    # is shared → COW into a private copy).
    cache.add_prefix(seq_id=1, prefix_tokens=[1, 2, 3, 4, 5, 6, 7], block_ids=table_owner)
    cache.stage_prefix(seq_id=2, prefix_block_ids=table_owner, num_prefix_tokens=6)
    last = torch.randn(1, 1, 2, 8, device=DEVICE, dtype=torch.float16)
    cache.update(seq_id=2, k_new=last, v_new=last, layer_idx=0)

    # If the owner wrote the SAME last-token K/V (idempotent rewrite, as the
    # model does), the staged seq matches the full prefill for every position.
    # With a differing write, the staged value is the caller's — the invariant
    # we assert is that the leading prefix positions still match the owner.
    # get() yields [num_kv_heads, num_tokens, head_dim]; transpose to
    # [num_tokens, num_kv_heads, head_dim] to compare against the prefill.
    k2, _ = cache.get(seq_id=2, start_idx=0, end_idx=7)
    assert torch.equal(k2.transpose(0, 1)[0:6], k_full[0, 0:6, :])
    # Token count advanced exactly to the full prompt length.
    assert cache.block_manager.get_num_tokens(2) == 7


def test_free_hitting_seq_preserves_owner_prefix_entry():
    """A sequence that REPLAYS the prefix must not evict the owner's prefix
    entry when it finishes: the cached blocks are still owned (pristine) by
    the registering sequence, so the entry remains valid for the next hit."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
        enable_prefix_cache=True,
    )
    k = torch.randn(1, 5, 2, 8, device=DEVICE)
    v = torch.randn(1, 5, 2, 8, device=DEVICE)
    table1 = cache.update(seq_id=1, k_new=k, v_new=v)
    tokens = [1, 2, 3, 4, 5]
    cache.add_prefix(seq_id=1, prefix_tokens=tokens, block_ids=table1)

    # seq 2 hits the prefix, extends, and frees — must NOT clear the entry.
    cache.stage_prefix(seq_id=2, prefix_block_ids=table1, num_prefix_tokens=4)
    cache.update(seq_id=2, k_new=torch.randn(1, 1, 2, 8, device=DEVICE), v_new=torch.randn(1, 1, 2, 8, device=DEVICE))
    cache.free(seq_id=2)
    assert cache.try_get_prefix_blocks(tokens) == table1, (
        "hitting sequence's free() must not evict the registering owner's entry"
    )

    # A third sequence can still replay the shared prefix.
    cache.stage_prefix(seq_id=3, prefix_block_ids=table1, num_prefix_tokens=4)

    # Freeing the OWNER (the registrar) drops the entry.
    cache.free(seq_id=1)
    assert cache.try_get_prefix_blocks(tokens) is None


def test_cow_on_shared_block_preserves_all_layers():
    """A COW'd block is a private copy of the FULL multi-layer block; later
    layers write into the remapped block and the owner's per-layer content is
    preserved on every layer."""
    cache = PagedKVCache(
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
        enable_prefix_cache=True,
    )
    for layer_idx in range(2):
        k = torch.randn(1, 5, 2, 8, device=DEVICE)
        v = torch.randn(1, 5, 2, 8, device=DEVICE)
        cache.update(seq_id=1, k_new=k, v_new=v, layer_idx=layer_idx)
    owner_before = [
        cache.get(seq_id=1, start_idx=0, end_idx=5, layer_idx=layer_idx)[0].clone() for layer_idx in range(2)
    ]
    table1 = cache.get_block_table(1)
    cache.add_prefix(seq_id=1, prefix_tokens=[1, 2, 3, 4, 5], block_ids=table1)

    cache.stage_prefix(seq_id=2, prefix_block_ids=table1, num_prefix_tokens=4)
    for layer_idx in range(2):
        k_new = torch.randn(1, 1, 2, 8, device=DEVICE)
        v_new = torch.randn(1, 1, 2, 8, device=DEVICE)
        cache.update(seq_id=2, k_new=k_new, v_new=v_new, layer_idx=layer_idx)

    # Owner's content is preserved on BOTH layers (the COW ran on layer 0 but
    # copied the full multi-layer block).
    for layer_idx in range(2):
        owner_after = cache.get(seq_id=1, start_idx=0, end_idx=5, layer_idx=layer_idx)[0]
        assert torch.equal(owner_before[layer_idx], owner_after)
    # seq 2's block table was remapped to a private boundary block on layer 0.
    assert cache.get_block_table(2)[1] != table1[1]


def test_add_prefix_snapshots_block_table_so_entry_does_not_grow():
    """REGRESSION (TASK-065 follow-up): ``add_prefix`` must snapshot the
    block table, not alias the owner's LIVE list. ``extend_sequence`` mutates
    the stored table in place as the owner decodes, so an aliased entry would
    silently grow past the prompt into the owner's decode blocks — a later
    hit would then fork the owner's whole decode table (pin + COW spiral)."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=16,
        block_size=4,
        device=DEVICE,
        enable_prefix_cache=True,
    )
    k = torch.randn(1, 5, 2, 8, device=DEVICE)
    v = torch.randn(1, 5, 2, 8, device=DEVICE)
    cache.update(seq_id=1, k_new=k, v_new=v)
    prompt_blocks = list(cache.get_block_table(1))  # snapshot (avoid aliasing)
    assert len(prompt_blocks) == 2

    cache.add_prefix(seq_id=1, prefix_tokens=[1, 2, 3, 4, 5], block_ids=cache.get_block_table(1))

    # The cached entry is a frozen snapshot of the 2 prompt blocks.
    entry = cache.prefix_cache.cache[cache._seq_to_hash[1]]
    assert list(entry) == prompt_blocks
    assert entry is not cache.get_block_table(1), "cached entry must not alias the live table"

    # Owner decodes past a block boundary (5 -> 10 tokens, 3 blocks). The
    # cached prompt entry must NOT grow into the decode blocks.
    for _ in range(5):
        cache.update(
            seq_id=1, k_new=torch.randn(1, 1, 2, 8, device=DEVICE), v_new=torch.randn(1, 1, 2, 8, device=DEVICE)
        )
    assert cache.block_manager.get_num_tokens(1) == 10
    assert list(cache.prefix_cache.cache[cache._seq_to_hash[1]]) == prompt_blocks, (
        "cached prefix entry grew alongside the owner's decode blocks (aliasing)"
    )

    # A hit stages ONLY the prompt blocks — a sibling request must not fork
    # the owner's decode block.
    staged = cache.try_get_prefix_blocks([1, 2, 3, 4, 5])
    assert staged == prompt_blocks
    table_hit = cache.stage_prefix(seq_id=2, prefix_block_ids=staged, num_prefix_tokens=4)
    assert table_hit == prompt_blocks
    owner_table = cache.get_block_table(1)
    assert not set(table_hit) - set(owner_table[:2]), "hit forked the owner's decode blocks"


def test_stage_prefix_full_hit_block_aligned_boundary():
    """Block-aligned full-prompt hit: the rewrite lands at the last token of a
    fully-filled block; the shared boundary block is COW'd and the owner's
    content is preserved (TASK-065 boundary case N % block_size == 0)."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=DEVICE,
        enable_prefix_cache=True,
    )
    k = torch.randn(1, 8, 2, 8, device=DEVICE, dtype=torch.float16)
    v = torch.randn(1, 8, 2, 8, device=DEVICE, dtype=torch.float16)
    table1 = cache.update(seq_id=1, k_new=k, v_new=v)
    owner_full = cache.get(seq_id=1, start_idx=0, end_idx=8)[0].clone()
    cache.add_prefix(seq_id=1, prefix_tokens=[1, 2, 3, 4, 5, 6, 7, 8], block_ids=table1)
    assert len(table1) == 2

    # Full hit: stage 7 prefix tokens then append the 8th at offset 7 (the
    # last slot of the shared block 1) -> COW.
    cache.stage_prefix(seq_id=2, prefix_block_ids=table1, num_prefix_tokens=7)
    last = torch.randn(1, 1, 2, 8, device=DEVICE, dtype=torch.float16)
    table2 = cache.update(seq_id=2, k_new=last, v_new=last, layer_idx=0)

    assert table2[0] == table1[0]
    assert table2[1] != table1[1]
    assert torch.equal(cache.get(seq_id=1, start_idx=0, end_idx=8)[0], owner_full)
    assert cache.block_manager.get_num_tokens(2) == 8
    # Staged seq kept the shared block-0 prefix and its own private block-1.
    k2, _ = cache.get(seq_id=2, start_idx=0, end_idx=8)
    assert torch.equal(k2.transpose(0, 1)[0:7], k[0, 0:7, :])


def test_free_of_evicted_stale_registrar_does_not_steal_reregistered_entry():
    """REGRESSION (TASK-065 follow-up): when a sequence's prefix entry is
    LRU-evicted and ANOTHER sequence later re-registers the same prompt hash,
    the first sequence's ``free()`` must not evict the second's live entry —
    only the current owner of the cached blocks may invalidate them."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=32,
        block_size=4,
        device=DEVICE,
        enable_prefix_cache=True,
        max_prefixes=2,
    )
    prompt_tokens = [1, 2, 3, 4, 5]

    def _prefill(seq_id, prompt):
        k = torch.randn(1, 5, 2, 8, device=DEVICE)
        v = torch.randn(1, 5, 2, 8, device=DEVICE)
        table = cache.update(seq_id=seq_id, k_new=k, v_new=v)
        cache.add_prefix(seq_id=seq_id, prefix_tokens=prompt, block_ids=table)
        return table

    # A registers the prompt; two DISTINCT prompts fill the LRU and evict A's.
    table_a = _prefill(1, prompt_tokens)
    assert cache.try_get_prefix_blocks(prompt_tokens) == table_a
    _prefill(2, [9, 8, 7, 6, 5]), _prefill(3, [4, 3, 2, 1, 0])
    assert cache.try_get_prefix_blocks(prompt_tokens) is None, "A's entry should be LRU-evicted"

    # B (a fresh prefill of the same prompt) re-registers the hash with its
    # own blocks — B is now the entry owner.
    table_b = _prefill(4, prompt_tokens)
    assert cache.try_get_prefix_blocks(prompt_tokens) == table_b

    # A finishes: its stale _seq_to_hash must not steal B's live entry.
    cache.free(1)
    assert cache.try_get_prefix_blocks(prompt_tokens) == table_b, (
        "stale registrar's free() stole the re-registered owner's entry"
    )

    # A third request (a hit, sharing B's blocks) finishing still must not
    # evict B's entry either.
    cache.stage_prefix(seq_id=5, prefix_block_ids=table_b, num_prefix_tokens=4)
    cache.free(5)
    assert cache.try_get_prefix_blocks(prompt_tokens) == table_b

    # Only the current owner's free() drops the entry.
    cache.free(4)
    assert cache.try_get_prefix_blocks(prompt_tokens) is None
