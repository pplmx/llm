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

```
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
    
    def clear(self) -> None:
        """Clear all cached prefixes."""
        self.cache.clear()
```

### 2. Integration with PagedKVCache

**Modified**: `src/llm/core/paged_attention/paged_kv_cache.py`

```python
class PagedKVCache:
    def __init__(self, ..., enable_prefix_cache: bool = False):
        # Existing init
        self.enable_prefix_cache = enable_prefix_cache
        if enable_prefix_cache:
            self.prefix_cache = PrefixCache()
    
    def add_prefix(self, prefix_tokens: list[int], block_ids: list[int]) -> None:
        """Cache prefix blocks for later reuse."""
        if not self.enable_prefix_cache:
            return
        
        prefix_hash = self._hash_tokens(prefix_tokens)
        self.prefix_cache.add(prefix_hash, block_ids)
    
    def get_prefix_block_ids(self, prefix_tokens: list[int]) -> list[int] | None:
        """Get cached block IDs if prefix matches."""
        if not self.enable_prefix_cache:
            return None
        
        prefix_hash = self._hash_tokens(prefix_tokens)
        return self.prefix_cache.get(prefix_hash)
    
    @staticmethod
    def _hash_tokens(tokens: list[int]) -> str:
        """Simple hash for token list."""
        return hash(tuple(tokens))
```

### 3. Integration with Attention Forward

```python
def paged_attention_forward(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    seq_lens: Tensor,
    num_kv_heads: int,
    prefix_cache: PrefixCache | None = None,  # NEW
    prefix_tokens: list[int] | None = None,     # NEW
    block_size: int = 16,
) -> Tensor:
    # Check cache first
    if prefix_cache and prefix_tokens is not None:
        cached_blocks = prefix_cache.get(hash_tokens(prefix_tokens))
        if cached_blocks:
            # Use cached blocks instead of computing
            ...
```

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

| Case | Handling |
|------|----------|
| Cache miss | Normal computation, no cache added |
| Cache full | LRU eviction of oldest entry |
| Partial match | Not supported (full prefix match only) |
| Different tokenizers | Hash is tokenizer-specific |

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