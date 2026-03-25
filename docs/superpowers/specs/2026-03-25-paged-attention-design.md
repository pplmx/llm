# Paged Attention Implementation Design

**Date**: 2026-03-25  
**Status**: Draft  
**Owner**: LLM Project

## Overview

Implement vLLM-style Paged Attention for memory-efficient inference. This enables dynamic memory allocation for KV cache, reducing memory waste and supporting longer contexts.

## Goals

- Implement block-level KV cache storage
- Add paged attention forward pass
- Support configurable block size (default: 16)
- Maximum context length: 4096 tokens
- Memory reduction: 50%+ compared to contiguous cache

## Architecture

```
src/llm/core/
├── kv_cache.py              # Existing KVCache (contiguous)
├── paged_attention/
│   ├── __init__.py
│   ├── block_allocator.py  # Existing: physical block allocation
│   ├── block_manager.py    # Existing: logical→physical mapping
│   ├── paged_kv_cache.py   # NEW: block-level KV storage
│   └── attention.py        # NEW: paged attention forward
```

## Components

### 1. PagedKVCache

**File**: `src/llm/core/paged_attention/paged_kv_cache.py`

```python
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
    ):
        # [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
        self.k_cache = torch.zeros(
            num_layers, num_blocks, num_kv_heads, block_size, head_dim,
            device=device, dtype=dtype
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        
        self.block_manager = BlockManager(num_blocks, block_size, num_layers)
        
    def update(self, seq_id: int, k_new: Tensor, v_new: Tensor) -> tuple[Tensor, Tensor]:
        """Append new tokens to sequence and return views into cache."""
        # 1. Allocate blocks if needed
        # 2. Copy k_new, v_new to cache
        # 3. Return views for attention computation
        pass
        
    def get(self, seq_id: int, start_idx: int, end_idx: int) -> tuple[Tensor, Tensor]:
        """Get KV cache slice for a sequence range."""
        pass
        
    def free(self, seq_id: int):
        """Free blocks when sequence completes."""
        self.block_manager.free_sequence(seq_id)
```

### 2. PagedAttention Forward

**File**: `src/llm/core/paged_attention/attention.py`

```python
def paged_attention_forward(
    q: Tensor,              # [batch, num_heads, 1, head_dim]
    k_cache: Tensor,        # [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
    v_cache: Tensor,        # Same shape as k_cache
    block_tables: Tensor,   # [batch, max_blocks] - physical block IDs
    seq_lens: Tensor,       # [batch] - current sequence lengths
    block_size: int = 16,
) -> Tensor:
    """Paged attention forward pass.
    
    Args:
        q: Query tensor for current token(s)
        k_cache: KV cache with block-level storage
        v_cache: Same as k_cache
        block_tables: Logical→physical block mapping
        seq_lens: Current sequence lengths
        
    Returns:
        Attention output tensor
    """
    # 1. For each query, gather K/V from scattered blocks
    # 2. Compute attention with FlashAttention or standard matmul
    # 3. Return output
    pass
```

### 3. Integration with Model

**Option**: Create a wrapper that switches between contiguous and paged:

```python
# In DecoderModel or attention module
def forward(self, hidden_states, use_paged_attention=False, **kwargs):
    if use_paged_attention:
        return self.paged_attention_forward(hidden_states, **kwargs)
    return self.standard_attention_forward(hidden_states, **kwargs)
```

## Block Allocation Strategy

### First-Come-First-Served (FCFS)

Simple allocation:
1. New sequence gets free blocks from pool
2. Blocks are never moved once allocated
3. On completion, blocks return to pool

### Memory Layout

```
Physical Memory (GPU):
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Block 0    │ Block 1    │ Block 2    │ Block 3    │
│ [2,8,16,64]│ [2,8,16,64]│ [2,8,16,64]│ [2,8,16,64]│
└─────────────┴─────────────┴─────────────┴─────────────┘

Logical Sequence (seq_1):
Tokens: [0, 1, 2, ... 47] → 3 blocks
Block Table: [0, 1, 2]
```

## Configuration

```python
# Serving config
LLM_SERVING_USE_PAGED_ATTENTION=true
LLM_SERVING_MAX_BLOCKS=256      # Total blocks (4096 tokens with block_size=16)
LLM_SERVING_BLOCK_SIZE=16
```

## Testing Strategy

1. **Unit tests**:
   - Block allocation/deallocation
   - KV cache update/get
   - Edge cases (empty sequence, full memory)

2. **Integration tests**:
   - Forward pass correctness vs standard attention
   - Memory usage comparison
   - Batch processing with multiple sequences

3. **E2E tests**:
   - Generate text with paged attention
   - Compare output with standard attention

## Implementation Order

1. PagedKVCache storage class
2. Block-level KV update/get
3. PagedAttention forward
4. Integration with DecoderModel
5. Config and optional enable
6. Tests

## Dependencies

No new dependencies. Uses existing:
- torch
- BlockAllocator (existing)
- BlockManager (existing)

## Backward Compatibility

- Default: `use_paged_attention=False`
- Existing inference code works unchanged
- Optional feature for memory-constrained scenarios