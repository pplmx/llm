"""
Paged Attention module for memory-efficient KV cache management.

Implements vLLM-style paged attention with block-based memory allocation.
"""

from llm.core.paged_attention.attention import paged_attention_forward
from llm.core.paged_attention.block_allocator import BlockAllocator
from llm.core.paged_attention.block_manager import BlockManager
from llm.core.paged_attention.paged_kv_cache import PagedKVCache

__all__ = ["BlockAllocator", "BlockManager", "PagedKVCache", "paged_attention_forward"]
