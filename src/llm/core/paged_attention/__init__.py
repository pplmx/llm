"""
Paged Attention module for memory-efficient KV cache management.

Implements vLLM-style paged attention with block-based memory allocation.
"""

from llm.core.paged_attention.block_allocator import BlockAllocator
from llm.core.paged_attention.block_manager import BlockManager

__all__ = ["BlockAllocator", "BlockManager"]
