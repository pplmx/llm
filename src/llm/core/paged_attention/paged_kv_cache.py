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

    def remove(self, prefix_hash: str) -> None:
        """Drop a prefix entry (no-op if absent).

        Used when the sequence that owned the cached blocks is freed — the
        stored block IDs become dangling and must not be replayed (RIL
        ISS-071).  Because LRU eviction may already have dropped the entry,
        absence is not an error.
        """
        self.cache.pop(prefix_hash, None)


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
        # Hash -> the sequence that currently OWNS the cached entry. ``free``
        # may only remove an entry whose current owner is the freed sequence:
        # a stale ``_seq_to_hash`` leftover from a sequence whose entry was
        # LRU-evicted (or overwritten by another sequence registering the same
        # prompt) must not steal a live entry registered by someone else.
        self._hash_to_owner: dict[str, int] = {}

    def _hash_tokens(self, tokens: list[int]) -> str:
        """Generate hash for token list.

        Uses ``array.array('i', ...)`` so token ids outside ``[0, 256)``
        (the norm for BPE/SentencePiece vocabularies) are handled correctly.
        """
        return hashlib.sha256(array.array("i", tokens).tobytes()).hexdigest()

    def add_prefix(self, seq_id: int, prefix_tokens: list[int], block_ids: list[int]) -> None:
        """Add prefix blocks to cache, owned by ``seq_id``.

        A sequence registers at most ONE live prefix entry: :meth:`free`
        drops the entry its owner registered (and only that entry), so a
        re-registration under a different hash evicts the previous one here
        to keep that bookkeeping exact (RIL TASK-065).
        """
        if not self.enable_prefix_cache or self.prefix_cache is None:
            return
        prefix_hash = self._hash_tokens(prefix_tokens)
        prev_hash = self._seq_to_hash.pop(seq_id, None)
        if prev_hash is not None and prev_hash != prefix_hash:
            self.prefix_cache.remove(prev_hash)
        # Snapshot, don't alias: the caller (the engine) passes the sequence's
        # LIVE block table, and the prefix must describe only the PREFIX's
        # blocks. Storing the list by reference would let the owner's later
        # ``extend_sequence`` grow the cached entry past the prompt into its
        # decode blocks; a subsequent hit would then fork the owner's WHOLE
        # decode table, pinning those blocks and forcing the owner into a
        # copy-on-write spiral on every decode step (RIL TASK-065 follow-up).
        self.prefix_cache.add(prefix_hash, list(block_ids))
        self._seq_to_hash[seq_id] = prefix_hash
        # The LAST registrar of a hash is the entry's owner (a fresh prefill
        # that re-encountered an LRU-evicted prompt re-registers its own
        # blocks). ``free`` consults this to avoid cross-sequence eviction.
        self._hash_to_owner[prefix_hash] = seq_id
        # Prune owner bookkeeping for hashes no longer cached (LRU eviction
        # drops the entry but not this map); keeps it bounded by max_prefixes.
        for stale in [k for k in self._hash_to_owner if k not in self.prefix_cache.cache]:
            del self._hash_to_owner[stale]

    def try_get_prefix_blocks(self, prefix_tokens: list[int]) -> list[int] | None:
        """Try to get cached prefix blocks."""
        if not self.enable_prefix_cache or self.prefix_cache is None:
            return None
        prefix_hash = self._hash_tokens(prefix_tokens)
        return self.prefix_cache.get(prefix_hash)

    def stage_prefix(self, seq_id: int, prefix_block_ids: list[int], num_prefix_tokens: int) -> list[int]:
        """Start a new sequence sharing a cached prefix (prefix replay).

        Creates the sequence in the block manager with the cached blocks
        forked into its table (shared, refcounted — no K/V is copied). The
        sequence is recorded as already owning ``num_prefix_tokens`` tokens,
        so a subsequent :meth:`update` appends at the right offset and copies
        on write before a token lands inside a still-shared block.

        Args:
            seq_id: Sequence identifier (slot id on the engine path).
            prefix_block_ids: Block ids from :meth:`try_get_prefix_blocks`.
            num_prefix_tokens: Number of already-owned prefix tokens.

        Returns:
            The staged block table.
        """
        if not self.enable_prefix_cache:
            raise RuntimeError("stage_prefix requires enable_prefix_cache=True")
        return self.block_manager.allocate_sequence_shared_prefix(seq_id, prefix_block_ids, num_prefix_tokens)

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
            # Re-fetch the LIVE table (``allocate_sequence`` returns a copy)
            # so a layer-0 COW remap below propagates to ``get_block_table``
            # callers instead of vanishing into the throwaway copy.
            self.block_manager.allocate_sequence(seq_id, num_new_tokens)
            block_table = self.block_manager.get_block_table(seq_id)
            start_token_offset = 0

        # Write the new tokens into the (possibly extended) block table.
        # Each new token goes into the block whose relative index matches
        # ``(start_token_offset + i) // block_size``.  The leading index is
        # ``layer_idx`` (NOT ``:``) — ``:`` would broadcast that layer's K/V
        # into every layer's slice, so a multi-layer decoder attends over the
        # wrong K/V for every layer but the one that wrote last.
        #
        # Layer 0 owns the block-table remapping: when a staged-prefix
        # sequence (or any shared-block holder) writes into a block that is
        # still referenced by the prefix-cache owner, it must copy-on-write
        # first — writing in place would corrupt the cached K/V the owner
        # (and the model's attention) still reads (RIL TASK-065). Later
        # layers read the remapped table fresh from the manager, so they
        # land in the private block without re-running the COW.
        for i in range(num_new_tokens):
            global_pos = start_token_offset + i
            block_idx = global_pos // self.block_size
            in_block_offset = global_pos % self.block_size
            block_id = block_table[block_idx]
            if layer_idx == 0:
                block_id = self._copy_on_write_if_shared(block_id, block_table, block_idx)
            self.k_cache[layer_idx, block_id, :, in_block_offset, :] = k_transposed[:, :, i, :]
            self.v_cache[layer_idx, block_id, :, in_block_offset, :] = v_transposed[:, :, i, :]

        return block_table

    def _copy_on_write_if_shared(self, block_id: int, block_table: list[int], block_idx: int) -> int:
        """Private-copy ``block_id`` before an in-place write if it is shared.

        A block is shared when another sequence is still reading it — the
        prefix-cache owner whose blocks a staged sequence references, or a
        sibling sequence forked from the same prefix. Overwriting it in place
        would leak the write into the other sequence's context. The fresh
        block receives the FULL multi-layer content (all transformer layers
        share one physical block) before the write, and the sequence's block
        table is remapped; the original block is left byte-identical for its
        other readers.

        Returns the id to write into (unchanged when not shared).
        """
        if not self.block_manager.is_block_shared(block_id):
            return block_id
        old_id = block_id
        new_id = self.block_manager.cow_block(old_id)
        # Copy the preserved prefix content from the shared block before this
        # layer's write overwrites its slice; later layers read the same
        # logical block and must see the full history (TASK-065).
        with torch.no_grad():
            self.k_cache[:, new_id, :, :, :].copy_(self.k_cache[:, old_id, :, :, :])
            self.v_cache[:, new_id, :, :, :].copy_(self.v_cache[:, old_id, :, :, :])
        block_table[block_idx] = new_id
        return new_id

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
        """Free blocks when sequence completes.

        The sequence's OWN prefix-cache entry (the one it registered via
        :meth:`add_prefix`) is dropped BEFORE its blocks are freed: the entry
        stores this sequence's physical block IDs, and once
        ``free_sequence`` returns them to the allocator a later request may
        be handed the same blocks. Leaving it in place would replay
        another/in-flight sequence's newly-written K/V as a cached prefix —
        use-after-free of the KV blocks (RIL ISS-071).

        Only the entry the sequence itself registered is removed (via
        ``_seq_to_hash``, which :meth:`add_prefix` keeps exact by evicting a
        stale prior entry on re-registration). A sequence that merely
        REPLAYED a prefix via :meth:`stage_prefix` shared the *owner's*
        blocks, which remain live (and pristine — every write into a shared
        block copy-on-writes) for as long as the owner holds them, so its
        free must not evict an entry it does not own; doing so would make the
        first replay destroy the very cache entry that served it (RIL
        TASK-065).
        """
        if self.prefix_cache is not None:
            prefix_hash = self._seq_to_hash.pop(seq_id, None)
            # Only the CURRENT owner of the entry may remove it. A stale
            # ``_seq_to_hash`` mapping from a sequence whose entry was
            # LRU-evicted, or superseded by another sequence's registration of
            # the same prompt, must not evict a live entry owned by another
            # still-running sequence (cross-owner theft).
            if prefix_hash is not None and self._hash_to_owner.get(prefix_hash) == seq_id:
                self.prefix_cache.remove(prefix_hash)
                self._hash_to_owner.pop(prefix_hash, None)
        self.block_manager.free_sequence(seq_id)
