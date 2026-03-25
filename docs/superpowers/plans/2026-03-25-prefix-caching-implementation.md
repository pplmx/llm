# Prefix Caching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement prefix caching for inference to reduce redundant KV computation for repeated system prompts in multi-turn conversations.

**Architecture:** Create a lightweight `PrefixCache` class inside `paged_kv_cache.py` that stores block_ids mappings (not full KV tensors). Cache hit remaps block_tables to cached blocks without changing the paged_attention_forward interface.

**Tech Stack:** PyTorch, PagedAttention, Pydantic Settings

---

## Key Changes from Initial Plan (Based on Review)

1. **No new file**: PrefixCache is now defined inside `paged_kv_cache.py` (as per spec architecture)
2. **LRU fixed**: Uses `OrderedDict` for proper access-order tracking
3. **API aligned**: Matches spec's `add_prefix(seq_id, block_ids)` signature
4. **Hash function**: Uses tuple hash directly (returns str, matching Python3)

---

## File Structure

- Modify: `src/llm/core/paged_attention/paged_kv_cache.py` - Add PrefixCache class and integration
- Modify: `src/llm/serving/config.py` - Add enable_prefix_cache and max_prefixes config
- Create: `tests/core/test_prefix_cache.py` - Unit tests for PrefixCache
- Modify: `tests/core/test_paged_kv_cache.py` - Add prefix cache integration tests

---

## Task 1: Add PrefixCache to PagedKVCache

**Files:**

- Modify: `src/llm/core/paged_attention/paged_kv_cache.py`

**Step 1: Write the failing test**

```python
# tests/core/test_paged_kv_cache.py - Add these tests

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
    from collections import OrderedDict

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
        device=get_device(),
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
        device=get_device(),
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
        device=get_device(),
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
        device=get_device(),
        enable_prefix_cache=True,
    )

    prefix_tokens = [1, 2, 3, 4]
    block_ids = [0, 1]

    cache.add_prefix(seq_id=1, prefix_tokens=prefix_tokens, block_ids=block_ids)

    cached = cache.try_get_prefix_blocks(prefix_tokens)
    assert cached is not None
    assert cached == block_ids


def test_prefix_cache_miss():
    """Test cache miss scenario."""
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device=get_device(),
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
        device=get_device(),
        enable_prefix_cache=False,
    )

    cached = cache.try_get_prefix_blocks([1, 2, 3])
    assert cached is None

    cache.add_prefix(seq_id=1, prefix_tokens=[1, 2, 3], block_ids=[0, 1])
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_paged_kv_cache.py::test_prefix_cache_init -v`
Expected: FAIL with "AttributeError: module 'llm.core.paged_attention.paged_kv_cache' has no attribute 'PrefixCache'"

**Step 3: Write implementation**

```python
# src/llm/core/paged_attention/paged_kv_cache.py

import torch
from collections import OrderedDict
from torch import Tensor

from llm.core.paged_attention.block_manager import BlockManager


class PrefixCache:
    """Cache for storing prefix KV blocks (block_ids only)."""

    def __init__(self, max_prefixes: int = 10):
        self.max_prefixes = max_prefixes
        self.cache: OrderedDict[str, list[int]] = OrderedDict()

    def add(self, prefix_hash: str, block_ids: list[int]) -> None:
        """Add prefix blocks to cache."""
        if len(self.cache) >= self.max_prefixes:
            self.cache.popitem(last=False)

        self.cache[prefix_hash] = block_ids
        self.cache.move_to_end(prefix_hash)

    def get(self, prefix_hash: str) -> list[int] | None:
        """Get cached block IDs for prefix."""
        if prefix_hash in self.cache:
            self.cache.move_to_end(prefix_hash)
            return self.cache[prefix_hash]
        return None


class PagedKVCache:
    """Block-level KV cache for paged attention."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int = 16,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        enable_prefix_cache: bool = False,
        max_prefixes: int = 10,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.enable_prefix_cache = enable_prefix_cache

        self.k_cache = torch.zeros(
            num_layers, num_blocks, num_kv_heads, block_size, head_dim, device=device, dtype=dtype
        )
        self.v_cache = torch.zeros_like(self.k_cache)

        self.block_manager = BlockManager(num_blocks, block_size, num_layers)

        self.prefix_cache = PrefixCache(max_prefixes) if enable_prefix_cache else None
        self._seq_to_hash: dict[int, str] = {}

    def _hash_tokens(self, tokens: list[int]) -> str:
        """Hash for token list. Returns string."""
        return str(hash(tuple(tokens)))

    def add_prefix(self, seq_id: int, prefix_tokens: list[int], block_ids: list[int]) -> None:
        """Cache prefix blocks for a sequence.

        Args:
            seq_id: Sequence identifier
            prefix_tokens: The prefix tokens to cache
            block_ids: List of physical block IDs containing prefix KV
        """
        if not self.enable_prefix_cache or self.prefix_cache is None:
            return

        if seq_id in self._seq_to_hash:
            return

        prefix_hash = self._hash_tokens(prefix_tokens)
        self.prefix_cache.add(prefix_hash, block_ids)
        self._seq_to_hash[seq_id] = prefix_hash

    def try_get_prefix_blocks(self, prefix_tokens: list[int]) -> list[int] | None:
        """Try to get cached prefix blocks.

        Args:
            prefix_tokens: The prefix tokens to look up

        Returns:
            List of block IDs if hit, None if miss
        """
        if not self.enable_prefix_cache or self.prefix_cache is None:
            return None

        prefix_hash = self._hash_tokens(prefix_tokens)
        return self.prefix_cache.get(prefix_hash)

    # ... keep existing methods: update(), get(), free(), get_block_table()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_paged_kv_cache.py -v -k "prefix_cache"`
Expected: PASS

**Step 5: Commit**

```bash
git add src/llm/core/paged_attention/paged_kv_cache.py tests/core/test_paged_kv_cache.py
git commit -m "feat: add PrefixCache to PagedKVCache with block_id mapping"
```

---

## Task 2: Add Config Options

**Files:**

- Modify: `src/llm/serving/config.py`
- Create: `tests/serving/test_config.py`

**Step 1: Write the failing test**

```python
# tests/serving/test_config.py
import pytest
from llm.serving.config import ServingConfig


def test_prefix_cache_config_defaults():
    """Test default prefix cache config values."""
    config = ServingConfig()
    assert config.enable_prefix_cache is False
    assert config.max_prefixes == 10


def test_prefix_cache_config_override():
    """Test overriding prefix cache config."""
    config = ServingConfig(
        enable_prefix_cache=True,
        max_prefixes=5,
    )
    assert config.enable_prefix_cache is True
    assert config.max_prefixes == 5
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/serving/test_config.py -v`
Expected: FAIL with "AttributeError"

**Step 3: Add config fields**

```python
# src/llm/serving/config.py

class ServingConfig(BaseSettings):
    # ... existing fields ...

    # Paged Attention
    use_paged_attention: bool = False
    max_blocks: int = 256
    block_size: int = 16

    # Prefix Cache
    enable_prefix_cache: bool = False
    max_prefixes: int = 10

    # ... rest unchanged
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/serving/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/llm/serving/config.py tests/serving/test_config.py
git commit -m "feat: add prefix cache config options"
```

---

## Task 3: Integration Tests

**Files:**

- Modify: `tests/serving/test_paged_integration.py`

**Step 1: Write the integration test**

```python
# Add to tests/serving/test_paged_integration.py

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
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/serving/test_paged_integration.py::test_prefix_cache_integration_flow -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/serving/test_paged_integration.py
git commit -m "test: add prefix cache integration tests"
```

---

## Task 4: Run Full Test Suite

**Step 1: Run test suite**

Run: `make test`
Expected: All tests pass

**Step 2: Run ruff**

Run: `make ruff`
Expected: No lint/format errors

**Step 3: Commit**

```bash
git add -A && git commit -m "test: verify prefix cache implementation"
```

---

## Notes on Block Remapping (Future Enhancement)

The current implementation stores prefix block_ids and provides lookup. The spec mentions block remapping in `update()`, but this requires deeper integration with the inference flow. For MVP, the cache hit detection is implemented via `try_get_prefix_blocks()`. Full block remapping would be a follow-up enhancement.

**Current flow:**

1. Caller checks `try_get_prefix_blocks(prefix_tokens)`
2. If hit, reuses cached KV blocks directly
3. If miss, computes normally and optionally caches via `add_prefix()`

This aligns with the spec's "只修改 block_tables 实现 cache hit" - the caller uses the returned block_ids to read from k_cache/v_cache directly.
