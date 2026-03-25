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

```text
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

    def update(self, seq_id: int, k_new: Tensor, v_new: Tensor) -> list[int]:
        """Append new tokens to sequence.

        Returns:
            List of physical block IDs allocated for this sequence (block_table).
            Caller uses this to populate block_tables tensor.

        Raises:
            RuntimeError: If no free blocks available (OOM).
        """
        # Check if blocks available, raise if not (OOM handling)
        num_tokens = k_new.shape[1]
        if not self.block_manager.can_allocate_sequence(num_tokens):
            raise RuntimeError("No free blocks available for new sequence")

        # Allocate blocks via block_manager and copy KV to cache
        block_ids = self.block_manager.allocate_sequence(seq_id, num_tokens)

        # Copy k_new, v_new to the allocated blocks in cache
        # For each block, copy the corresponding tokens
        for i, block_id in enumerate(block_ids):
            start_token = i * self.block_size
            end_token = min(start_token + self.block_size, num_tokens)
            self.k_cache[:, block_id, :, :end_token-start_token, :] = k_new[:, :, start_token:end_token, :]
            self.v_cache[:, block_id, :, :end_token-start_token, :] = v_new[:, :, start_token:end_token, :]

        return block_ids

    def get_block_table(self, seq_id: int) -> list[int]:
        """Get block table (physical block IDs) for a sequence."""
        return self.block_manager.get_block_table(seq_id)

    def get(self, seq_id: int, start_idx: int, end_idx: int) -> tuple[Tensor, Tensor]:
        """Get KV cache slice for a sequence range.

        Returns:
            k: [num_kv_heads, num_tokens, head_dim]
            v: [num_kv_heads, num_tokens, head_dim]
        """
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
    block_tables: Tensor,   # [batch, max_blocks] - physical block IDs for each sequence
    seq_lens: Tensor,       # [batch] - current sequence lengths
    block_size: int = 16,
) -> Tensor:
    """Paged attention forward pass.

    Args:
        q: Query tensor for current token(s)
        k_cache: KV cache with block-level storage
        v_cache: Same as k_cache
        block_tables: [batch, max_blocks] physical block IDs for each sequence.
                      Obtained from PagedKVCache.get_block_table() after each update().
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

**Data Flow**:

1. Caller (e.g., ContinuousBatchingEngine) allocates sequence via `paged_kv_cache.update(seq_id, k, v)`
2. Returns `list[int]` - block IDs for this sequence
3. Caller maintains `block_tables: dict[str, list[int]]` - request_id → block IDs
4. When calling forward, caller builds `block_tables` tensor:

   ```python
   # batch_size = len(requests)
   block_tables = torch.zeros(batch_size, max_blocks, dtype=torch.long)
   for i, req in enumerate(requests):
       block_tables[i, :len(self.block_map[req.id])] = torch.tensor(self.block_map[req.id])
   ```

5. Pass `block_tables` to `paged_attention_forward()`

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

### Out-of-Memory Handling

**Strategy**: Fail-fast for simplicity

- When a new sequence requires blocks but none are free:
    - `update()` raises `RuntimeError("No free blocks available")`
    - Scheduler catches this and handles (preemption or queue waiting)
    - This is delegated to the higher-level Scheduler/Engine

**Note**: Future enhancement could include:

- Block eviction (oldest sequence)
- Waiting queue (async generation)
- Memory defragmentation

### Memory Layout

```text
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
