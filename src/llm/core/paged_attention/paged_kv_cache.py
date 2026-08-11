"""Paged KV Cache for memory-efficient inference."""

import array
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
        """Generate hash for token list.

        Uses ``array.array('i', ...)`` so token ids outside ``[0, 256)``
        (the norm for BPE/SentencePiece vocabularies) are handled correctly.
        """
        return hashlib.sha256(array.array("i", tokens).tobytes()).hexdigest()

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

    def update(self, seq_id: int, k_new: Tensor, v_new: Tensor, layer_idx: int = 0) -> list[int]:
        """Append new tokens to sequence.

        For a brand-new sequence this allocates fresh blocks; for an
        existing sequence it extends the block table only if the new
        tokens cross a block boundary.

        ``layer_idx`` scopes the write to a single transformer layer: the
        decoder calls :meth:`update` once per layer with that layer's own K/V,
        and each call must write **only** its own slice of ``k_cache`` /
        ``v_cache`` (which are ``[num_layers, ...]``).  Only the layer that
        calls first (``layer_idx == 0``) allocates / extends the block table
        and advances the sequence's token count; the remaining layers reuse
        the same block table and write at the same token offsets without
        re-advancing.  (Default 0 keeps the single-layer contract intact.)

        Args:
            seq_id: Sequence identifier.
            k_new: [batch, tokens, num_kv_heads, head_dim]
            v_new: [batch, tokens, num_kv_heads, head_dim]
            layer_idx: Which layer's cache slice to write.

        Returns:
            List of physical block IDs allocated for this sequence
            (initial allocation) or the current full block table
            (extension).
        """
        num_new_tokens = k_new.shape[1]
        k_transposed = k_new.transpose(1, 2)
        v_transposed = v_new.transpose(1, 2)

        if seq_id in self.block_manager.sequences:
            if layer_idx == 0:
                # First layer of this step extends / accounts the sequence.
                existing_num_tokens = self.block_manager.get_num_tokens(seq_id)
                block_table = self.block_manager.extend_sequence(seq_id, num_new_tokens)
                start_token_offset = existing_num_tokens
            else:
                # Later layers already saw layer 0 extend the table this
                # step; they write the same new tokens without re-advancing
                # the token count (which lives on the shared block manager).
                block_table = self.get_block_table(seq_id)
                start_token_offset = self.block_manager.get_num_tokens(seq_id) - num_new_tokens
        else:
            if layer_idx != 0:
                raise RuntimeError(
                    f"Sequence {seq_id} has no block table yet; layer 0 must update() before layer {layer_idx}."
                )
            if not self.block_manager.can_allocate_sequence(num_new_tokens):
                raise RuntimeError("No free blocks available for new sequence")
            block_table = self.block_manager.allocate_sequence(seq_id, num_new_tokens)
            start_token_offset = 0

        # Write the new tokens into the (possibly extended) block table.
        # Each new token goes into the block whose relative index matches
        # ``(start_token_offset + i) // block_size``.  The leading index is
        # ``layer_idx`` (NOT ``:``) — ``:`` would broadcast that layer's K/V
        # into every layer's slice, so a multi-layer decoder attends over the
        # wrong K/V for every layer but the one that wrote last.
        for i in range(num_new_tokens):
            global_pos = start_token_offset + i
            block_idx = global_pos // self.block_size
            in_block_offset = global_pos % self.block_size
            block_id = block_table[block_idx]
            self.k_cache[layer_idx, block_id, :, in_block_offset, :] = k_transposed[:, :, i, :]
            self.v_cache[layer_idx, block_id, :, in_block_offset, :] = v_transposed[:, :, i, :]

        return block_table

    def get_block_table(self, seq_id: int) -> list[int]:
        """Get block IDs for a sequence."""
        return self.block_manager.get_block_table(seq_id)

    def get(self, seq_id: int, start_idx: int, end_idx: int, layer_idx: int = 0) -> tuple[Tensor, Tensor]:
        """Get KV cache slice for a sequence range.

        Args:
            seq_id: Sequence identifier.
            start_idx: Starting token index (inclusive).
            end_idx: Ending token index (exclusive).
            layer_idx: Which layer's cache slice to read (default 0 keeps
                the single-layer contract; multi-layer readers must pass the
                layer whose KV they're attending over).

        Raises:
            ValueError: If ``start_idx`` or ``end_idx`` are out of bounds
                or ``start_idx >= end_idx``.
        """
        block_table = self.get_block_table(seq_id)

        num_tokens = self.block_manager.get_num_tokens(seq_id)
        if start_idx < 0 or end_idx > num_tokens:
            raise ValueError(
                f"Index range [{start_idx}:{end_idx}] out of bounds for sequence {seq_id} with {num_tokens} tokens"
            )
        if start_idx >= end_idx:
            raise ValueError(f"start_idx ({start_idx}) must be less than end_idx ({end_idx})")

        k_seq = []
        v_seq = []

        start_block = start_idx // self.block_size
        end_block = (end_idx - 1) // self.block_size + 1

        for block_id in block_table[start_block:end_block]:
            # Index on the layer axis ``layer_idx``, not ``:`` — the cache is
            # ``[num_layers, num_blocks, ...]`` and each layer read/writes only
            # its own slice.  After the layer slice each block is
            # ``[num_kv_heads, block_size, head_dim]`` so blocks concatenate
            # along dim 1 (the token axis).
            k_seq.append(self.k_cache[layer_idx, block_id, :, : self.block_size, :])
            v_seq.append(self.v_cache[layer_idx, block_id, :, : self.block_size, :])

        k_full = torch.cat(k_seq, dim=1)
        v_full = torch.cat(v_seq, dim=1)

        start_offset = start_idx % self.block_size
        num_tokens = end_idx - start_idx

        # ``k_full`` is ``[num_kv_heads, num_tokens, head_dim]`` (the layer
        # axis was consumed by the ``layer_idx`` slice above).
        return k_full[:, start_offset : start_offset + num_tokens, :], v_full[
            :, start_offset : start_offset + num_tokens, :
        ]

    def free(self, seq_id: int):
        """Free blocks when sequence completes."""
        self.block_manager.free_sequence(seq_id)
