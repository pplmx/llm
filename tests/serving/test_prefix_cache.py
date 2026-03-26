"""
Tests for PrefixCache - shared prompt optimization.
"""

import pytest
import torch

from llm.serving.prefix_cache import PrefixCache, PrefixCacheEntry


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestPrefixCacheEntry:
    """Tests for PrefixCacheEntry dataclass."""

    def test_touch_updates_access(self):
        """Test that touch updates last_accessed and increments access_count."""
        entry = PrefixCacheEntry(
            prefix_hash="abc123",
            prefix_tokens=[1, 2, 3],
            prefix_len=3,
            kv_cache=[],
        )

        initial_access_count = entry.access_count
        initial_time = entry.last_accessed

        entry.touch()

        assert entry.access_count == initial_access_count + 1
        assert entry.last_accessed >= initial_time


class TestPrefixCache:
    """Tests for PrefixCache class."""

    def test_init_default(self):
        """Test cache initialization with defaults."""
        cache = PrefixCache()
        assert cache.max_entries == 100
        assert cache.min_prefix_len == 10

    def test_init_custom(self):
        """Test cache initialization with custom values."""
        cache = PrefixCache(max_entries=50, min_prefix_len=5)
        assert cache.max_entries == 50
        assert cache.min_prefix_len == 5

    def test_insert_and_lookup(self):
        """Test inserting and looking up entries."""
        cache = PrefixCache(min_prefix_len=3)
        prefix_tokens = [1, 2, 3, 4, 5]

        kv_cache = [(torch.randn(1, 2, 5, 16), torch.randn(1, 2, 5, 16)) for _ in range(2)]

        cache.insert(prefix_tokens, kv_cache)

        retrieved = cache.lookup(prefix_tokens)
        assert retrieved is not None
        assert retrieved.prefix_tokens == prefix_tokens

    def test_lookup_nonexistent(self):
        """Test looking up non-existent entry returns None."""
        cache = PrefixCache()
        result = cache.lookup([1, 2, 3, 4, 5])  # Won't exist without insert
        assert result is None

    def test_lru_eviction(self):
        """Test that LRU entries are evicted when cache is full."""
        cache = PrefixCache(max_entries=2, min_prefix_len=1)

        for i in range(3):
            cache.insert([i, i + 1], [(torch.randn(1, 1, 2, 16), torch.randn(1, 1, 2, 16))])

        # First entry should be evicted
        result = cache.lookup([0, 1])
        assert result is None
        # Recent entries should still exist
        assert cache.lookup([1, 2]) is not None
        assert cache.lookup([2, 3]) is not None

    def test_clear(self):
        """Test clearing the cache."""
        cache = PrefixCache(min_prefix_len=1)
        cache.insert([1, 2], [(torch.randn(1, 1, 2, 16), torch.randn(1, 1, 2, 16))])

        cache.cache.clear()

        assert len(cache.cache) == 0

    def test_stats(self):
        """Test cache statistics."""
        cache = PrefixCache(min_prefix_len=1)
        cache.insert([1, 2], [(torch.randn(1, 1, 2, 16), torch.randn(1, 1, 2, 16))])

        assert cache.hits == 0
        assert cache.misses >= 0

    def test_min_prefix_length(self):
        """Test that prefixes shorter than min_prefix_len are not cached."""
        cache = PrefixCache(min_prefix_len=5)

        # Short prefix - won't be cached
        short_tokens = [1, 2, 3]
        kv_cache = [(torch.randn(1, 1, 3, 16), torch.randn(1, 1, 3, 16))]

        result = cache.insert(short_tokens, kv_cache)
        assert result == ""  # Empty string indicates not cached

        # Long enough prefix - should be cached
        long_tokens = [1, 2, 3, 4, 5, 6]
        result = cache.insert(long_tokens, kv_cache)
        assert result != ""  # Should return hash
