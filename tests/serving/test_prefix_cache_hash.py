"""Tests for ``SlotPrefixCache.hash_tokens`` with realistic token ids.

The prefix-cache key is derived from the prompt token ids via SHA-256.
The original implementation used ``bytes(tokens)``, which only works when
every token id is in ``[0, 256)``.  Real tokenizers (BPE, SentencePiece)
have vocabularies of 50k-128k, so ids routinely exceed 255 — causing a
``ValueError`` that would crash the prefix-cache feature when enabled.

These tests verify the hash function accepts ids across the full ``int``
range and is deterministic / collision-free for distinct token lists.
"""

from llm.serving.batch_engine import SlotPrefixCache


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
