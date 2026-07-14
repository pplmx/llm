"""Paged KV Cache for memory-efficient inference."""

import hashlib
from collections import OrderedDict

import torch
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

        self.k_cache = torch.zeros(
            num_layers, num_blocks, num_kv_heads, block_size, head_dim, device=device, dtype=dtype
        )
        self.v_cache = torch.zeros_like(self.k_cache)

        self.block_manager = BlockManager(num_blocks, block_size, num_layers)

        self.enable_prefix_cache = enable_prefix_cache
        self.prefix_cache = PrefixCache(max_prefixes) if enable_prefix_cache else None
        self._seq_to_hash: dict[int, str] = {}

    def _hash_tokens(self, tokens: list[int]) -> str:
        """Generate hash for token list."""
        return hashlib.sha256(bytes(tokens)).hexdigest()

    def add_prefix(self, seq_id: int, prefix_tokens: list[int], block_ids: list[int]) -> None:
        """Add prefix blocks to cache."""
        if not self.enable_prefix_cache or self.prefix_cache is None:
            return
        prefix_hash = self._hash_tokens(prefix_tokens)
        self.prefix_cache.add(prefix_hash, block_ids)
        self._seq_to_hash[seq_id] = prefix_hash

    def try_get_prefix_blocks(self, prefix_tokens: list[int]) -> list[int] | None:
        """Try to get cached prefix blocks."""
        if not self.enable_prefix_cache or self.prefix_cache is None:
            return None
        prefix_hash = self._hash_tokens(prefix_tokens)
        return self.prefix_cache.get(prefix_hash)

    def update(self, seq_id: int, k_new: Tensor, v_new: Tensor) -> list[int]:
        """Append new tokens to sequence.

        For a brand-new sequence this allocates fresh blocks; for an
        existing sequence it extends the block table only if the new
        tokens cross a block boundary.

        Args:
            seq_id: Sequence identifier.
            k_new: [batch, tokens, num_kv_heads, head_dim]
            v_new: [batch, tokens, num_kv_heads, head_dim]

        Returns:
            List of physical block IDs allocated for this sequence
            (initial allocation) or the current full block table
            (extension).
        """
        num_new_tokens = k_new.shape[1]
        k_transposed = k_new.transpose(1, 2)
        v_transposed = v_new.transpose(1, 2)

        if seq_id in self.block_manager.sequences:
            # Extend an existing sequence.
            existing_num_tokens = self.block_manager.get_num_tokens(seq_id)
            block_table = self.block_manager.extend_sequence(seq_id, num_new_tokens)
            start_token_offset = existing_num_tokens
        else:
            if not self.block_manager.can_allocate_sequence(num_new_tokens):
                raise RuntimeError("No free blocks available for new sequence")
            block_table = self.block_manager.allocate_sequence(seq_id, num_new_tokens)
            start_token_offset = 0

        # Write the new tokens into the (possibly extended) block table.
        # Each new token goes into the block whose relative index matches
        # ``(start_token_offset + i) // block_size``.
        for i in range(num_new_tokens):
            global_pos = start_token_offset + i
            block_idx = global_pos // self.block_size
            in_block_offset = global_pos % self.block_size
            block_id = block_table[block_idx]
            self.k_cache[:, block_id, :, in_block_offset, :] = k_transposed[:, :, i, :]
            self.v_cache[:, block_id, :, in_block_offset, :] = v_transposed[:, :, i, :]

        return block_table

    def get_block_table(self, seq_id: int) -> list[int]:
        """Get block IDs for a sequence."""
        return self.block_manager.get_block_table(seq_id)

    def get(self, seq_id: int, start_idx: int, end_idx: int) -> tuple[Tensor, Tensor]:
        """Get KV cache slice for a sequence range."""
        block_table = self.get_block_table(seq_id)

        k_seq = []
        v_seq = []

        start_block = start_idx // self.block_size
        end_block = (end_idx - 1) // self.block_size + 1

        for block_id in block_table[start_block:end_block]:
            k_seq.append(self.k_cache[:, block_id, :, : self.block_size, :])
            v_seq.append(self.v_cache[:, block_id, :, : self.block_size, :])

        k_full = torch.cat(k_seq, dim=2)
        v_full = torch.cat(v_seq, dim=2)

        start_offset = start_idx % self.block_size
        num_tokens = end_idx - start_idx

        return k_full[0, :, start_offset : start_offset + num_tokens, :], v_full[
            0, :, start_offset : start_offset + num_tokens, :
        ]

    def free(self, seq_id: int):
        """Free blocks when sequence completes."""
        self.block_manager.free_sequence(seq_id)
