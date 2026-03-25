# Paged Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement vLLM-style Paged Attention for memory-efficient inference with block-level KV cache.

**Architecture:** 
- Block-level KV storage instead of contiguous memory
- BlockManager already exists, need to implement PagedKVCache and PagedAttention forward
- Optional integration via config switch

**Tech Stack:** Python, PyTorch

---

## File Structure

```
src/llm/core/paged_attention/
├── __init__.py                    # Existing, add exports
├── paged_kv_cache.py             # NEW: PagedKVCache class
├── attention.py                  # NEW: paged_attention_forward
├── block_allocator.py            # Existing
└── block_manager.py              # Existing
```

---

## Task 1: PagedKVCache Storage Class

**Files:**
- Create: `src/llm/core/paged_attention/paged_kv_cache.py`
- Test: `tests/core/test_paged_kv_cache.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_paged_kv_cache.py
import pytest
import torch
from llm.core.paged_attention.paged_kv_cache import PagedKVCache

def test_paged_kv_cache_init():
    cache = PagedKVCache(
        num_layers=2,
        num_kv_heads=4,
        head_dim=16,
        num_blocks=32,
        block_size=16,
        device="cpu",
        dtype=torch.float16,
    )
    assert cache.k_cache.shape == (2, 32, 4, 16, 16)

def test_update_allocates_blocks():
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device="cpu",
    )
    k = torch.randn(1, 2, 4, 8)  # [batch, heads, tokens, head_dim]
    v = torch.randn(1, 2, 4, 8)
    
    block_ids = cache.update(seq_id=1, k_new=k, v_new=v)
    
    assert len(block_ids) == 1  # 4 tokens = 1 block
    assert block_ids == [0]

def test_free_sequence():
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=8,
        block_size=4,
        device="cpu",
    )
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    
    cache.update(seq_id=1, k_new=k, v_new=v)
    assert cache.block_manager.num_free_blocks == 7
    
    cache.free(seq_id=1)
    assert cache.block_manager.num_free_blocks == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_paged_kv_cache.py -v`
Expected: FAIL with "No module named 'llm.core.paged_attention.paged_kv_cache'"

- [ ] **Step 3: Write PagedKVCache implementation**

```python
# src/llm/core/paged_attention/paged_kv_cache.py
"""Paged KV Cache for memory-efficient inference."""

import torch
from torch import Tensor

from llm.core.paged_attention.block_manager import BlockManager


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
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        
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
            List of physical block IDs allocated for this sequence.
        """
        num_tokens = k_new.shape[2]  # [batch, heads, tokens, head_dim]
        
        if not self.block_manager.can_allocate_sequence(num_tokens):
            raise RuntimeError("No free blocks available for new sequence")
        
        block_ids = self.block_manager.allocate_sequence(seq_id, num_tokens)
        
        # Copy KV to cache blocks
        for i, block_id in enumerate(block_ids):
            start_token = i * self.block_size
            end_token = min(start_token + self.block_size, num_tokens)
            num_tokens_in_block = end_token - start_token
            
            self.k_cache[:, block_id, :, :num_tokens_in_block, :] = k_new[:, :, start_token:end_token, :]
            self.v_cache[:, block_id, :, :num_tokens_in_block, :] = v_new[:, :, start_token:end_token, :]
        
        return block_ids
        
    def get_block_table(self, seq_id: int) -> list[int]:
        """Get block IDs for a sequence."""
        return self.block_manager.get_block_table(seq_id)
        
    def free(self, seq_id: int):
        """Free blocks when sequence completes."""
        self.block_manager.free_sequence(seq_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_paged_kv_cache.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/core/paged_attention/paged_kv_cache.py tests/core/test_paged_kv_cache.py
git commit -m "feat(inference): add PagedKVCache storage class"
```

---

## Task 2: PagedAttention Forward

**Files:**
- Create: `src/llm/core/paged_attention/attention.py`
- Test: `tests/core/test_paged_attention.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_paged_attention.py
import pytest
import torch
from llm.core.paged_attention.attention import paged_attention_forward


def test_paged_attention_output_shape():
    batch_size = 2
    num_heads = 4
    head_dim = 16
    block_size = 16
    num_kv_heads = 2
    num_blocks = 8
    
    # Query for current token
    q = torch.randn(batch_size, num_heads, 1, head_dim)
    
    # KV cache: [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
    k_cache = torch.randn(1, num_blocks, num_kv_heads, block_size, head_dim)
    v_cache = torch.randn(1, num_blocks, num_kv_heads, block_size, head_dim)
    
    # Block tables: [batch, max_blocks]
    block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    seq_lens = torch.tensor([20, 25])  # tokens in each sequence
    
    output = paged_attention_forward(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        num_kv_heads=num_kv_heads,
    )
    
    assert output.shape == (batch_size, num_heads, 1, head_dim)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_paged_attention.py -v`
Expected: FAIL with "No module named 'llm.core.paged_attention.attention'"

- [ ] **Step 3: Write implementation**

```python
# src/llm/core/paged_attention/attention.py
"""Paged Attention forward implementation."""

import torch
from torch import Tensor


def paged_attention_forward(
    q: Tensor,              # [batch, num_heads, 1, head_dim]
    k_cache: Tensor,        # [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
    v_cache: Tensor,        # Same shape as k_cache
    block_tables: Tensor,   # [batch, max_blocks] - physical block IDs
    seq_lens: Tensor,       # [batch] - current sequence lengths
    num_kv_heads: int,
    block_size: int = 16,
) -> Tensor:
    """Paged attention forward pass.
    
    Args:
        q: Query tensor for current token(s)
        k_cache: KV cache with block-level storage
        v_cache: Same as k_cache
        block_tables: [batch, max_blocks] physical block IDs for each sequence
        seq_lens: Current sequence lengths
        num_kv_heads: Number of KV heads
        block_size: Tokens per block
        
    Returns:
        Attention output tensor [batch, num_heads, 1, head_dim]
    """
    batch_size, num_heads, _, head_dim = q.shape
    num_layers, num_blocks, _, _, _ = k_cache.shape
    
    # For now, support single-layer attention
    k_cache = k_cache[0]  # [num_blocks, num_kv_heads, block_size, head_dim]
    v_cache = v_cache[0]
    
    # Gather KV from blocks for each sequence
    k_gathered = []
    v_gathered = []
    
    for b in range(batch_size):
        seq_len = seq_lens[b].item()
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        
        # Get physical block IDs for this sequence
        seq_block_ids = block_tables[b, :num_blocks_needed].tolist()
        
        # Gather KV from these blocks
        k_seq = []
        v_seq = []
        for i, block_id in enumerate(seq_block_ids):
            start = 0
            if i == num_blocks_needed - 1:
                # Last block: only valid tokens
                end = seq_len - (num_blocks_needed - 1) * block_size
            else:
                end = block_size
                
            k_seq.append(k_cache[block_id, :, start:end, :])
            v_seq.append(v_cache[block_id, :, start:end, :])
        
        k_gathered.append(torch.cat(k_seq, dim=1))  # [num_kv_heads, seq_len, head_dim]
        v_gathered.append(torch.cat(v_seq, dim=1))
    
    # Stack for batch
    k_full = torch.stack(k_gathered, dim=0)  # [batch, num_kv_heads, seq_len, head_dim]
    v_full = torch.stack(v_gathered, dim=0)
    
    # Expand KV heads if num_kv_heads != num_heads (for GQA)
    if num_kv_heads != num_heads:
        repeat_factor = num_heads // num_kv_heads
        k_full = k_full.repeat_interleave(repeat_factor, dim=1)
        v_full = v_full.repeat_interleave(repeat_factor, dim=1)
    
    # Attention: Q @ K^T * scale
    scale = head_dim ** -0.5
    attn_weights = torch.matmul(q, k_full.transpose(-2, -1)) * scale
    
    # Softmax
    attn_weights = torch.softmax(attn_weights, dim=-1)
    
    # Output: attn_weights @ V
    output = torch.matmul(attn_weights, v_full)
    
    return output
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_paged_attention.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/core/paged_attention/attention.py tests/core/test_paged_attention.py
git commit -m "feat(inference): add paged attention forward"
```

---

## Task 3: Integration with Config

**Files:**
- Modify: `src/llm/serving/config.py`
- Modify: `src/llm/core/paged_attention/__init__.py`

- [ ] **Step 1: Add config option**

```python
# In ServingConfig, add:
use_paged_attention: bool = False
max_blocks: int = 256
```

- [ ] **Step 2: Export PagedKVCache**

```python
# In __init__.py, add:
from llm.core.paged_attention.paged_kv_cache import PagedKVCache
from llm.core.paged_attention.attention import paged_attention_forward

__all__ = [..., "PagedKVCache", "paged_attention_forward"]
```

- [ ] **Step 3: Commit**

```bash
git add src/llm/serving/config.py src/llm/core/paged_attention/__init__.py
git commit -m "feat(inference): add paged attention config and exports"
```

---

## Task 4: Integration Test with ContinuousBatchingEngine

**Files:**
- Test: `tests/serving/test_paged_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/serving/test_paged_integration.py
import pytest
import torch
from llm.core.paged_attention.paged_kv_cache import PagedKVCache
from llm.core.paged_attention.attention import paged_attention_forward


def test_end_to_end_paged_inference():
    """Test paged attention workflow with multiple sequences."""
    # Setup
    cache = PagedKVCache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=16,
        num_blocks=32,
        block_size=16,
        device="cpu",
    )
    
    # Simulate two sequences
    seq1_k = torch.randn(1, 2, 10, 16)
    seq1_v = torch.randn(1, 2, 10, 16)
    block_ids_1 = cache.update(seq_id=1, k_new=seq1_k, v_new=seq1_v)
    
    seq2_k = torch.randn(1, 2, 8, 16)
    seq2_v = torch.randn(1, 2, 8, 16)
    block_ids_2 = cache.update(seq_id=2, k_new=seq2_k, v_new=seq2_v)
    
    # Build block_tables tensor
    block_tables = torch.zeros(2, 2, dtype=torch.long)
    block_tables[0, :len(block_ids_1)] = torch.tensor(block_ids_1)
    block_tables[1, :len(block_ids_2)] = torch.tensor(block_ids_2)
    seq_lens = torch.tensor([10, 8])
    
    # Forward pass
    q = torch.randn(2, 4, 1, 16)
    output = paged_attention_forward(
        q=q,
        k_cache=cache.k_cache,
        v_cache=cache.v_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        num_kv_heads=2,
    )
    
    assert output.shape == (2, 4, 1, 16)
    
    # Cleanup
    cache.free(seq_id=1)
    cache.free(seq_id=2)
```

- [ ] **Step 2: Run test**

Run: `pytest tests/serving/test_paged_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/serving/test_paged_integration.py
git commit -m "test(inference): add paged attention integration test"
```

---

## Summary

This plan implements Paged Attention in 4 tasks:

1. **PagedKVCache** - Block-level KV storage
2. **PagedAttention Forward** - Attention computation with block tables
3. **Config Integration** - Add config options and exports
4. **Integration Test** - E2E test with multiple sequences

Each task follows TDD: write failing test → implement → verify pass → commit.