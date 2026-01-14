"""Tests for advanced serving features."""

import time

import pytest
import torch

from llm.serving.prefix_cache import PrefixCache, RadixPrefixCache
from llm.serving.priority_scheduler import Priority, PriorityScheduler
from llm.serving.schemas import Sequence


class TestPriorityScheduler:
    """Tests for PriorityScheduler."""

    @pytest.fixture
    def scheduler(self):
        """Create a priority scheduler."""
        return PriorityScheduler(max_batch_size=4, enable_aging=False)

    @pytest.fixture
    def mock_sequence(self):
        """Create a mock sequence factory."""

        def make_seq(request_id: str) -> Sequence:
            return Sequence(
                request_id=request_id,
                prompt="test",
                input_ids=[1, 2, 3],
            )

        return make_seq

    def test_add_sequence(self, scheduler, mock_sequence):
        """Test adding sequences to scheduler."""
        seq = mock_sequence("req1")
        scheduler.add_sequence(seq, priority=Priority.NORMAL)

        assert scheduler.num_waiting == 1
        assert scheduler.queues[Priority.NORMAL]

    def test_priority_ordering(self, scheduler, mock_sequence):
        """Test that high priority runs before low priority."""
        low = mock_sequence("low")
        high = mock_sequence("high")

        scheduler.add_sequence(low, priority=Priority.LOW)
        scheduler.add_sequence(high, priority=Priority.HIGH)

        running = scheduler.schedule()

        # High priority should be scheduled first
        assert running[0].request_id == "high"
        assert running[1].request_id == "low"

    def test_same_priority_fifo(self, scheduler, mock_sequence):
        """Test FIFO within same priority."""
        first = mock_sequence("first")
        second = mock_sequence("second")

        scheduler.add_sequence(first, priority=Priority.NORMAL)
        scheduler.add_sequence(second, priority=Priority.NORMAL)

        running = scheduler.schedule()

        assert running[0].request_id == "first"
        assert running[1].request_id == "second"

    def test_max_batch_size(self, scheduler, mock_sequence):
        """Test batch size limit is respected."""
        for i in range(10):
            scheduler.add_sequence(mock_sequence(f"req{i}"))

        running = scheduler.schedule()

        assert len(running) == 4  # max_batch_size
        assert scheduler.num_waiting == 6

    def test_get_sequence(self, scheduler, mock_sequence):
        """Test finding sequences by ID."""
        seq = mock_sequence("target")
        scheduler.add_sequence(seq)

        found = scheduler.get_sequence("target")
        assert found is not None
        assert found.request_id == "target"

        not_found = scheduler.get_sequence("nonexistent")
        assert not_found is None

    def test_queue_stats(self, scheduler, mock_sequence):
        """Test queue statistics."""
        scheduler.add_sequence(mock_sequence("high"), Priority.HIGH)
        scheduler.add_sequence(mock_sequence("normal"), Priority.NORMAL)
        scheduler.schedule()

        stats = scheduler.get_queue_stats()

        assert stats["running"] == 2
        assert stats["total_scheduled"] == 2

    def test_clear(self, scheduler, mock_sequence):
        """Test clearing scheduler."""
        scheduler.add_sequence(mock_sequence("req1"))
        scheduler.schedule()

        scheduler.clear()

        assert not scheduler.has_pending_work
        assert scheduler.num_waiting == 0


class TestPrioritySchedulerAging:
    """Tests for scheduler aging mechanism."""

    def test_aging_boosts_priority(self):
        """Test that aging boosts priority of waiting requests."""
        scheduler = PriorityScheduler(
            max_batch_size=1,
            enable_aging=True,
            aging_interval_sec=0.0,  # Immediate aging
            aging_boost=1,
        )

        # Fill running slot
        first = Sequence(request_id="first", prompt="test", input_ids=[1])
        scheduler.add_sequence(first, Priority.HIGH)
        scheduler.schedule()

        # Add low priority that should age
        second = Sequence(request_id="second", prompt="test", input_ids=[1])
        scheduler.add_sequence(second, Priority.LOW)

        # Simulate time passing and schedule
        time.sleep(0.01)
        scheduler._apply_aging()

        # Should have been boosted
        assert scheduler.current_priorities.get("second") == Priority.NORMAL


class TestPrefixCache:
    """Tests for PrefixCache."""

    def test_insert_and_lookup(self):
        """Test basic insert and lookup."""
        cache = PrefixCache(min_prefix_len=3)

        tokens = [1, 2, 3, 4, 5]
        kv_cache = [(torch.randn(1, 4, 8), torch.randn(1, 4, 8))]

        cache.insert(tokens, kv_cache)
        entry = cache.lookup(tokens)

        assert entry is not None
        assert entry.prefix_tokens == tokens

    def test_lookup_miss(self):
        """Test cache miss."""
        cache = PrefixCache(min_prefix_len=3)

        entry = cache.lookup([1, 2, 3])

        assert entry is None
        assert cache.misses == 1

    def test_prefix_matching(self):
        """Test that prefix matching works."""
        cache = PrefixCache(min_prefix_len=3)

        prefix = [1, 2, 3, 4, 5]
        kv_cache = [(torch.randn(1, 4, 8), torch.randn(1, 4, 8))]
        cache.insert(prefix, kv_cache)

        # Longer sequence should match prefix
        longer = [1, 2, 3, 4, 5, 6, 7, 8]
        entry = cache.lookup(longer)

        assert entry is not None
        assert entry.prefix_len == 5

    def test_lru_eviction(self):
        """Test LRU eviction when at capacity."""
        cache = PrefixCache(max_entries=2, min_prefix_len=3)

        kv_cache = [(torch.randn(1, 4, 8), torch.randn(1, 4, 8))]

        cache.insert([1, 2, 3], kv_cache)
        cache.insert([4, 5, 6], kv_cache)
        cache.insert([7, 8, 9], kv_cache)  # Should evict first

        assert len(cache) == 2
        assert cache.lookup([1, 2, 3]) is None

    def test_stats(self):
        """Test cache statistics."""
        cache = PrefixCache(min_prefix_len=3)

        kv_cache = [(torch.randn(1, 4, 8), torch.randn(1, 4, 8))]
        cache.insert([1, 2, 3], kv_cache)

        cache.lookup([1, 2, 3])  # Hit
        cache.lookup([4, 5, 6])  # Miss

        stats = cache.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestRadixPrefixCache:
    """Tests for RadixPrefixCache."""

    def test_insert_and_lookup(self):
        """Test basic insert and lookup."""
        cache = RadixPrefixCache(min_prefix_len=1)

        tokens = [1, 2, 3, 4, 5]
        kv_cache = [(torch.randn(1, 4, 8), torch.randn(1, 4, 8))]

        cache.insert(tokens, kv_cache)
        matched_len, found_cache = cache.lookup(tokens)

        assert matched_len == 5
        assert found_cache is not None

    def test_longest_prefix_match(self):
        """Test finding longest matching prefix."""
        cache = RadixPrefixCache(min_prefix_len=1)

        kv_cache_short = [(torch.randn(1, 4, 8), torch.randn(1, 4, 8))]
        kv_cache_long = [(torch.randn(1, 4, 16), torch.randn(1, 4, 16))]

        cache.insert([1, 2, 3], kv_cache_short)
        cache.insert([1, 2, 3, 4, 5], kv_cache_long)

        # Should match longer prefix
        matched_len, _ = cache.lookup([1, 2, 3, 4, 5, 6, 7])
        assert matched_len == 5

    def test_no_match(self):
        """Test when no prefix matches."""
        cache = RadixPrefixCache(min_prefix_len=1)

        kv_cache = [(torch.randn(1, 4, 8), torch.randn(1, 4, 8))]
        cache.insert([1, 2, 3], kv_cache)

        matched_len, found_cache = cache.lookup([9, 8, 7])

        assert matched_len == 0
        assert found_cache is None

    def test_stats(self):
        """Test cache statistics."""
        cache = RadixPrefixCache(min_prefix_len=1)

        kv_cache = [(torch.randn(1, 4, 8), torch.randn(1, 4, 8))]
        cache.insert([1, 2, 3], kv_cache)

        cache.lookup([1, 2, 3])  # Hit
        cache.lookup([9, 8, 7])  # Miss

        stats = cache.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
