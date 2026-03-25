# Prefix Caching Implementation Design

**Date**: 2026-03-25  
**Status**: Draft  
**Owner**: LLM Project

## Overview

Implement prefix caching for inference to reduce redundant KV computation for repeated system prompts in multi-turn conversations.

## Goals

- Cache KV cache blocks for repeated prefixes (e.g., system prompts)
- Zero overhead when cache miss (fallback to normal computation)
- Simple in-memory storage (no persistence needed)

## Architecture

```text
src/llm/core/paged_attention/
├── paged_kv_cache.py      # Existing: PagedKVCache
├── prefix_cache.py        # NEW: PrefixCache class
└── ...
```

## Components

### 1. PrefixCache

**File**: `src/llm/core/paged_attention/prefix_cache.py`

```python
class PrefixCache:
    """Cache for storing prefix KV blocks."""

    def __init__(self, max_prefixes: int = 10):
        self.max_prefixes = max_prefixes
        self.cache: dict[str, list[int]] = {}  # hash -> block_ids

    def add(self, prefix_hash: str, block_ids: list[int]) -> None:
        """Add prefix blocks to cache.

        Args:
            prefix_hash: Hash of the prefix tokens
            block_ids: List of physical block IDs containing prefix KV
        """
        if len(self.cache) >= self.max_prefixes:
            # LRU eviction: remove oldest entry
            oldest = next(iter(self.cache))
            del self.cache[oldest]

        self.cache[prefix_hash] = block_ids

    def get(self, prefix_hash: str) -> list[int] | None:
        """Get cached block IDs for prefix.

        Args:
            prefix_hash: Hash of the prefix tokens

        Returns:
            List of block IDs if hit, None if miss
        """
        return self.cache.get(prefix_hash)


```

### 2. Integration with PagedKVCache

**Modified**: `src/llm/core/paged_attention/paged_kv_cache.py`

核心设计：**只修改 block_tables 实现 cache hit**，不改动 paged_attention_forward 接口

```python
class PagedKVCache:
    def __init__(self, ..., enable_prefix_cache: bool = False):
        # Existing init
        self.enable_prefix_cache = enable_prefix_cache
        self.prefix_cache = PrefixCache() if enable_prefix_cache else None
        self._seq_to_hash: dict[int, str] = {}  # seq_id -> hash

    def _hash_tokens(self, tokens: list[int]) -> int:
        """Simple hash for token list. Returns int."""
        return hash(tuple(tokens))

    def add_prefix(self, seq_id: int, block_ids: list[int]) -> None:
        """Cache prefix blocks for a sequence."""
        if not self.enable_prefix_cache or self.prefix_cache is None:
            return

        if seq_id in self._seq_to_hash:
            # Already cached
            return

        prefix_hash = self._hash_tokens(self.prefix_tokens)
        self.prefix_cache.add(prefix_hash, block_ids)
        self._seq_to_hash[seq_id] = prefix_hash

    def try_get_prefix_blocks(self, prefix_tokens: list[int]) -> list[int] | None:
        """Try to get cached prefix blocks. Returns block_ids if hit, None if miss."""
        if not self.enable_prefix_cache or self.prefix_cache is None:
            return None

        prefix_hash = self._hash_tokens(prefix_tokens)
        return self.prefix_cache.get(prefix_hash)

### 3. Usage Flow (Cache Hit 处理)

**逻辑**：在 PagedKVCache.update() 时检测 prefix 是否已缓存

```python
def update(self, seq_id, k_new, v_new, prefix_tokens=None):
    # 如果提供了 prefix_tokens，检查是否命中缓存
    if prefix_tokens:
        cached_blocks = self.try_get_prefix_blocks(prefix_tokens)
        if cached_blocks:
            # Cache hit: remap block_tables to cached blocks
            block_ids = cached_blocks  # Use cached blocks
            # 注意：需要处理 block_ids 和 seq_id 的映射关系
        else:
            # Cache miss: normal allocation
            block_ids = self.block_manager.allocate_sequence(...)
            # 并缓存新计算的 prefix
            if is_prefix:
                self.add_prefix(seq_id, block_ids)
    else:
        block_ids = self.block_manager.allocate_sequence(...)
```

**Block Remap 策略**：

- Cache hit 时，block_ids 直接指向 cached blocks
- 需要在 block_manager 中记录 (prefix_hash -> block_ids) 的映射
- 后续 attention 从 k_cache 读取时自动命中 cached blocks

## Usage Flow

```python
# 1. Initialize with prefix cache enabled
cache = PagedKVCache(
    num_layers=1,
    num_kv_heads=2,
    head_dim=16,
    num_blocks=256,
    enable_prefix_cache=True,
)

# 2. First request - compute and cache
prefix = tokenizer.encode("You are a helpful assistant.")
prefix_blocks = cache.update(seq_id=1, k=k_prefix, v=v_prefix)
cache.add_prefix(prefix, prefix_blocks)

# 3. Second request with same prefix - cache hit!
cached = cache.get_prefix_block_ids(prefix)
if cached:
    # Skip prefix computation, use cached blocks
    pass
```

## Configuration

```python
# ServingConfig
LLM_SERVING_ENABLE_PREFIX_CACHE=true
LLM_SERVING_MAX_PREFIXES=10  # Max cached prefixes
```

## Edge Cases

| Case                 | Handling                                                                               |
| -------------------- | -------------------------------------------------------------------------------------- |
| Cache miss           | Normal computation, no cache added                                                     |
| Cache full           | LRU eviction of oldest entry                                                           |
| Partial match        | Not supported (full prefix match only)                                                 |
| Multi-layer          | Per-layer: each layer maintains its own PrefixCache instance                           |
| Sequence completion  | Cached blocks are NOT freed when sequence completes - they are shared across sequences |
| Different tokenizers | Hash is tokenizer-specific                                                             |

## Implementation Order

1. PrefixCache class
2. Integrate with PagedKVCache
3. Integration with inference flow
4. Config options
5. Tests

## Testing Strategy

- Unit tests for hash collision resistance
- Cache hit/miss scenarios
- LRU eviction
- Integration with multi-sequence inference
