"""
Prefix Cache for shared prompt optimization.

Caches KV states for common prompt prefixes to avoid recomputation.
"""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class PrefixCacheEntry:
    """Cached prefix entry."""

    prefix_hash: str
    prefix_tokens: list[int]
    prefix_len: int
    kv_cache: list[tuple[torch.Tensor, torch.Tensor]]  # [(K, V) per layer]
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1


class PrefixCache:
    """
    LRU cache for KV states of common prompt prefixes.

    Enables reuse of computed KV states when multiple requests
    share the same prompt prefix (e.g., system prompts).
    """

    def __init__(
        self,
        max_entries: int = 100,
        max_memory_bytes: int | None = None,
        min_prefix_len: int = 10,
    ):
        """
        Initialize prefix cache.

        Args:
            max_entries: Maximum number of cached prefixes.
            max_memory_bytes: Optional memory limit in bytes.
            min_prefix_len: Minimum prefix length to cache.
        """
        self.max_entries = max_entries
        self.max_memory_bytes = max_memory_bytes
        self.min_prefix_len = min_prefix_len

        # LRU cache: prefix_hash -> PrefixCacheEntry
        self.cache: OrderedDict[str, PrefixCacheEntry] = OrderedDict()

        # Stats
        self.hits = 0
        self.misses = 0

    def get_prefix_hash(self, tokens: list[int]) -> str:
        """Compute hash for a token sequence."""
        token_bytes = bytes(tokens)
        return hashlib.sha256(token_bytes).hexdigest()[:16]

    def lookup(self, tokens: list[int]) -> PrefixCacheEntry | None:
        """
        Look up cached KV states for a token prefix.

        Tries to find the longest matching prefix.

        Args:
            tokens: Input token sequence.

        Returns:
            Cache entry if found, None otherwise.
        """
        if len(tokens) < self.min_prefix_len:
            self.misses += 1
            return None

        # Try to find longest matching prefix
        best_match: PrefixCacheEntry | None = None
        best_match_len = 0

        for entry in self.cache.values():
            prefix_len = entry.prefix_len

            if prefix_len > len(tokens):
                continue

            if prefix_len <= best_match_len:
                continue

            # Check if tokens match
            if tokens[:prefix_len] == entry.prefix_tokens:
                best_match = entry
                best_match_len = prefix_len

        if best_match:
            self.hits += 1
            best_match.touch()
            # Move to end (most recently used)
            self.cache.move_to_end(best_match.prefix_hash)
            return best_match
        else:
            self.misses += 1
            return None

    def insert(
        self,
        tokens: list[int],
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> str:
        """
        Insert a prefix and its KV states into cache.

        Args:
            tokens: Token sequence to cache.
            kv_cache: KV states per layer.

        Returns:
            Cache key (hash).
        """
        if len(tokens) < self.min_prefix_len:
            return ""

        prefix_hash = self.get_prefix_hash(tokens)

        # Update existing entry
        if prefix_hash in self.cache:
            self.cache[prefix_hash].touch()
            self.cache.move_to_end(prefix_hash)
            return prefix_hash

        # Evict if at capacity
        while len(self.cache) >= self.max_entries:
            self._evict_lru()

        # Check memory limit
        if self.max_memory_bytes:
            while self._get_memory_usage() >= self.max_memory_bytes and self.cache:
                self._evict_lru()

        # Insert new entry
        entry = PrefixCacheEntry(
            prefix_hash=prefix_hash,
            prefix_tokens=tokens.copy(),
            prefix_len=len(tokens),
            kv_cache=kv_cache,
        )
        self.cache[prefix_hash] = entry

        return prefix_hash

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self.cache:
            self.cache.popitem(last=False)

    def _get_memory_usage(self) -> int:
        """Estimate current memory usage in bytes."""
        total = 0
        for entry in self.cache.values():
            for k, v in entry.kv_cache:
                total += k.numel() * k.element_size()
                total += v.numel() * v.element_size()
        return total

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            "entries": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "memory_bytes": self._get_memory_usage(),
            "memory_mb": self._get_memory_usage() / (1024 * 1024),
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self.cache)

    def __contains__(self, tokens: list[int]) -> bool:
        prefix_hash = self.get_prefix_hash(tokens)
        return prefix_hash in self.cache


class RadixPrefixCache:
    """
    Radix tree-based prefix cache for efficient prefix matching.

    More efficient than hash-based cache for finding longest
    common prefixes across many cached sequences.
    """

    def __init__(
        self,
        max_nodes: int = 10000,
        min_prefix_len: int = 1,
    ):
        """
        Initialize radix prefix cache.

        Args:
            max_nodes: Maximum number of tree nodes.
            min_prefix_len: Minimum prefix length to cache.
        """
        self.max_nodes = max_nodes
        self.min_prefix_len = min_prefix_len

        # Root node
        self.root: dict[int, Any] = {"children": {}, "kv_cache": None, "count": 0}
        self.num_nodes = 1

        # Stats
        self.hits = 0
        self.misses = 0

    def lookup(self, tokens: list[int]) -> tuple[int, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """
        Find longest matching prefix in the radix tree.

        Args:
            tokens: Input tokens.

        Returns:
            Tuple of (matched_length, kv_cache or None).
        """
        node = self.root
        matched_len = 0
        best_kv_cache = None
        best_len = 0

        for i, token in enumerate(tokens):
            if token not in node["children"]:
                break

            node = node["children"][token]
            matched_len = i + 1

            if node["kv_cache"] is not None:
                best_kv_cache = node["kv_cache"]
                best_len = matched_len
                node["count"] += 1

        if best_kv_cache:
            self.hits += 1
        else:
            self.misses += 1

        return best_len, best_kv_cache

    def insert(
        self,
        tokens: list[int],
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> bool:
        """
        Insert a prefix and its KV cache into the tree.

        Args:
            tokens: Token sequence.
            kv_cache: KV cache per layer.

        Returns:
            True if inserted, False if at capacity.
        """
        if len(tokens) < self.min_prefix_len:
            return False

        node = self.root

        for token in tokens:
            if token not in node["children"]:
                if self.num_nodes >= self.max_nodes:
                    return False

                node["children"][token] = {"children": {}, "kv_cache": None, "count": 0}
                self.num_nodes += 1

            node = node["children"][token]

        node["kv_cache"] = kv_cache
        node["count"] += 1
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            "num_nodes": self.num_nodes,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
        }

    def clear(self) -> None:
        """Clear the tree."""
        self.root = {"children": {}, "kv_cache": None, "count": 0}
        self.num_nodes = 1
        self.hits = 0
        self.misses = 0
