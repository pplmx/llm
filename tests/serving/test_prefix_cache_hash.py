"""Tests for ``SlotPrefixCache.hash_tokens`` with realistic token ids.

The prefix-cache key is derived from the prompt token ids via SHA-256.
The original implementation used ``bytes(tokens)``, which only works when
every token id is in ``[0, 256)``.  Real tokenizers (BPE, SentencePiece)
have vocabularies of 50k-128k, so ids routinely exceed 255 — causing a
``ValueError`` that would crash the prefix-cache feature when enabled.

These tests verify the hash function accepts ids across the full ``int``
range and is deterministic / collision-free for distinct token lists.
"""

import pytest

from llm.serving.batch_engine import SlotAllocator, SlotPrefixCache


class TestHashTokensRealisticIds:
    """``hash_tokens`` must handle ids >= 256 (the common case for BPE)."""

    def test_hash_accepts_id_256(self):
        """Token id 256 (the first value that breaks ``bytes()``) must hash."""
        h = SlotPrefixCache.hash_tokens([1, 2, 3, 256])
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest

    def test_hash_accepts_large_vocab_ids(self):
        """GPT-2 vocabulary is 50 257 — every id below that must hash."""
        h = SlotPrefixCache.hash_tokens([50000, 50256, 0, 1])
        assert isinstance(h, str)

    def test_hash_accepts_single_large_id(self):
        h = SlotPrefixCache.hash_tokens([99999])
        assert isinstance(h, str)

    def test_hash_is_deterministic(self):
        """Same input → same hash."""
        ids = [100, 200, 300, 400]
        assert SlotPrefixCache.hash_tokens(ids) == SlotPrefixCache.hash_tokens(ids)

    def test_hash_differs_for_different_inputs(self):
        """Different token lists must produce different hashes (no collisions)."""
        h1 = SlotPrefixCache.hash_tokens([1, 2, 3, 4])
        h2 = SlotPrefixCache.hash_tokens([1, 2, 3, 5])
        assert h1 != h2

    def test_hash_differs_for_reordered_inputs(self):
        """Order matters — [1, 2] ≠ [2, 1]."""
        h1 = SlotPrefixCache.hash_tokens([1, 2, 3, 4])
        h2 = SlotPrefixCache.hash_tokens([2, 1, 3, 4])
        assert h1 != h2


class TestPrefixCachePutGetWithLargeIds:
    """The full put/get cycle must work with large token ids."""

    def test_put_and_get_round_trip_with_large_ids(self):
        cache = SlotPrefixCache(max_prefixes=10, min_prefix_len=4)
        tokens = [500, 1000, 2000, 50000]
        cache.put(tokens, slot=3, prefix_len=4)
        result = cache.get(tokens)
        assert result == (3, 4)

    def test_put_and_get_round_trip_with_mixed_ids(self):
        cache = SlotPrefixCache(max_prefixes=10, min_prefix_len=4)
        tokens = [1, 256, 50257, 0]
        cache.put(tokens, slot=0, prefix_len=4)
        result = cache.get(tokens)
        assert result == (0, 4)

    def test_get_returns_none_for_unknown_large_id_tokens(self):
        cache = SlotPrefixCache(max_prefixes=10, min_prefix_len=4)
        cache.put([500, 1000, 2000, 50000], slot=3, prefix_len=4)
        assert cache.get([501, 1000, 2000, 50000]) is None


class TestSlotPrefixCacheEviction:
    """LRU eviction and min_prefix_len behavior."""

    def test_lru_eviction_removes_oldest(self):
        """When the cache is full, the least-recently-used entry is evicted."""
        cache = SlotPrefixCache(max_prefixes=2, min_prefix_len=4)
        cache.put([1, 2, 3, 4], slot=0, prefix_len=4)
        cache.put([5, 6, 7, 8], slot=1, prefix_len=4)
        # Cache is full; adding a third should evict the first (LRU).
        cache.put([9, 10, 11, 12], slot=2, prefix_len=4)
        assert cache.get([1, 2, 3, 4]) is None
        assert cache.get([5, 6, 7, 8]) == (1, 4)
        assert cache.get([9, 10, 11, 12]) == (2, 4)

    def test_access_promotes_lru_order(self):
        """Accessing an entry should move it to the end (most-recently-used)."""
        cache = SlotPrefixCache(max_prefixes=2, min_prefix_len=4)
        cache.put([1, 2, 3, 4], slot=0, prefix_len=4)
        cache.put([5, 6, 7, 8], slot=1, prefix_len=4)
        # Access the first entry — it becomes MRU.
        cache.get([1, 2, 3, 4])
        # Adding a third should evict the second entry (now LRU).
        cache.put([9, 10, 11, 12], slot=2, prefix_len=4)
        assert cache.get([1, 2, 3, 4]) == (0, 4)
        assert cache.get([5, 6, 7, 8]) is None

    def test_put_below_min_prefix_len_is_noop(self):
        """Tokens shorter than min_prefix_len should not be cached."""
        cache = SlotPrefixCache(max_prefixes=10, min_prefix_len=4)
        cache.put([1, 2, 3], slot=0, prefix_len=3)
        assert cache.get([1, 2, 3]) is None

    def test_get_below_min_prefix_len_returns_none(self):
        """get() with too-short tokens returns None even if cached."""
        cache = SlotPrefixCache(max_prefixes=10, min_prefix_len=4)
        # Short tokens should never be cached.
        assert cache.get([1, 2, 3]) is None
        assert cache.get([]) is None

    def test_put_with_short_prefix_len_but_long_tokens(self):
        """put() checks token count, not prefix_len."""
        cache = SlotPrefixCache(max_prefixes=10, min_prefix_len=4)
        cache.put([1, 2, 3, 4, 5], slot=0, prefix_len=5)
        assert cache.get([1, 2, 3, 4, 5]) == (0, 5)

    def test_put_same_key_updates_not_evicts(self):
        """Re-putting the same tokens updates the value, not evicts."""
        cache = SlotPrefixCache(max_prefixes=2, min_prefix_len=4)
        cache.put([1, 2, 3, 4], slot=0, prefix_len=4)
        cache.put([1, 2, 3, 4], slot=5, prefix_len=4)
        assert cache.get([1, 2, 3, 4]) == (5, 4)
        assert len(cache._entries) == 1


class TestSlotAllocator:
    """Tests for SlotAllocator."""

    def test_allocate_returns_slot(self):
        allocator = SlotAllocator(total_slots=8)
        slot = allocator.allocate("req-1")
        assert isinstance(slot, int)
        assert 0 <= slot < 8
        assert allocator.num_free == 7

    def test_allocate_same_request_returns_same_slot(self):
        """Allocating the same request_id twice returns the same slot."""
        allocator = SlotAllocator(total_slots=8)
        slot1 = allocator.allocate("req-1")
        slot2 = allocator.allocate("req-1")
        assert slot1 == slot2

    def test_allocate_no_free_slots_raises(self):
        allocator = SlotAllocator(total_slots=2)
        allocator.allocate("req-1")
        allocator.allocate("req-2")
        with pytest.raises(RuntimeError, match="No free slots"):
            allocator.allocate("req-3")

    def test_free_returns_slot_to_pool(self):
        allocator = SlotAllocator(total_slots=8)
        allocator.allocate("req-1")
        assert allocator.num_free == 7
        allocator.free("req-1")
        assert allocator.num_free == 8

    def test_free_unknown_request_is_noop(self):
        allocator = SlotAllocator(total_slots=8)
        # Should not raise.
        allocator.free("nonexistent")
        assert allocator.num_free == 8

    def test_free_twice_is_noop(self):
        allocator = SlotAllocator(total_slots=8)
        allocator.allocate("req-1")
        allocator.free("req-1")
        # Second free should not raise or double-return the slot.
        allocator.free("req-1")
        assert allocator.num_free == 8

    def test_get_slot_returns_mapped_slot(self):
        allocator = SlotAllocator(total_slots=8)
        slot = allocator.allocate("req-1")
        assert allocator.get_slot("req-1") == slot

    def test_get_slot_returns_minus_one_for_unknown(self):
        allocator = SlotAllocator(total_slots=8)
        assert allocator.get_slot("nonexistent") == -1
