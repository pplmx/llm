"""
KV Cache manager for efficient autoregressive generation.

This module provides a pre-allocated cache that avoids repeated memory allocations
during token generation, significantly improving inference performance.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class KVCache:
    """Pre-allocated Key-Value cache for efficient autoregressive generation.

    Instead of using torch.cat to concatenate new K/V with past K/V (which allocates
    new memory each step), this class pre-allocates buffers and updates them in-place.

    Args:
        max_batch_size: Maximum batch size to support.
        max_seq_len: Maximum sequence length to cache.
        num_kv_heads: Number of key-value heads (for GQA, this may differ from num_heads).
        head_dim: Dimension of each attention head.
        device: Device to allocate buffers on.
        dtype: Data type for cache buffers.

    Example:
        >>> cache = KVCache(batch_size=2, max_seq_len=512, num_kv_heads=8, head_dim=64,
        ...                 device="cuda", dtype=torch.float16)
        >>> # In attention forward pass:
        >>> k, v = cache.update(k_new, v_new)  # In-place update, returns view
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Pre-allocate buffers: [B, N_kv, S_max, D]
        self.k_cache = torch.zeros(max_batch_size, num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(max_batch_size, num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        """Current cached sequence length."""
        return self._seq_len

    @property
    def device(self) -> torch.device:
        """Device of the cache buffers."""
        return self.k_cache.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the cache buffers."""
        return self.k_cache.dtype

    def update(self, k_new: Tensor, v_new: Tensor) -> tuple[Tensor, Tensor]:
        """Update cache with new key-value tensors and return full cache view.

        Args:
            k_new: New key tensor of shape [B, N_kv, S_new, D].
            v_new: New value tensor of shape [B, N_kv, S_new, D].

        Returns:
            Tuple of (k_cached, v_cached) containing all cached keys and values
            up to and including the new tokens. Shape: [B, N_kv, S_total, D].

        Raises:
            ValueError: If update would exceed max_seq_len.
        """
        batch_size = k_new.size(0)
        new_tokens = k_new.size(2)
        new_seq_len = self._seq_len + new_tokens

        if new_seq_len > self.max_seq_len:
            raise ValueError(
                f"Cache overflow: trying to cache {new_seq_len} tokens, but max_seq_len is {self.max_seq_len}"
            )

        # In-place update (no memory allocation)
        self.k_cache[:batch_size, :, self._seq_len : new_seq_len] = k_new
        self.v_cache[:batch_size, :, self._seq_len : new_seq_len] = v_new
        self._seq_len = new_seq_len

        # Return view of valid cache region
        return (
            self.k_cache[:batch_size, :, :new_seq_len],
            self.v_cache[:batch_size, :, :new_seq_len],
        )

    def update_at_indices(
        self,
        batch_indices: Tensor,
        k_new: Tensor,
        v_new: Tensor,
        start_pos: Tensor | int,
    ) -> tuple[Tensor, Tensor]:
        """Update cache at specific batch indices and positions.

        This is used for continuous batching or when batch slots are managed explicitly.

        Args:
            batch_indices: Tensor of shape [B_curr] containing the cache slot indices.
            k_new: New key tensor of shape [B_curr, N_kv, S_new, D].
            v_new: New value tensor of shape [B_curr, N_kv, S_new, D].
            start_pos: Starting position to write to. Can be an int (broadcast) or
                Tensor of shape ``[B_curr, S_new]`` (one position per batch slot
                and token). When a per-batch tensor is provided, the same value
                is written for every S_new position; we rely on the caller to
                pass position_ids so we never need a host-device ``.item()``
                sync on the happy path.

        Returns:
             Tuple of (k_out, v_out) for the current batch.
             Note: This returns the *full* valid context for the *current batch indices*.
             Shape: [B_curr, N_kv, Max_Context_Len, D].
             Since different sequences have different lengths, we return up to max(start_pos + S_new).
             Correct handling usually implies the model knows how to mask using attention mask.
        """
        seq_len_new = k_new.size(2)

        if isinstance(start_pos, int):
            # Scalar start_pos: every batch slot writes the same contiguous range.
            # This is the typical prefill path (all slots start at 0).
            pos_end = start_pos + seq_len_new
            if pos_end > self.max_seq_len:
                raise ValueError(
                    f"Cache overflow: trying to cache {pos_end} tokens, "
                    f"but max_seq_len is {self.max_seq_len}"
                )
            self.k_cache[batch_indices, :, start_pos:pos_end] = k_new
            self.v_cache[batch_indices, :, start_pos:pos_end] = v_new
        elif seq_len_new == 1:
            # Decode path: one position per batch slot, advanced indexing is
            # already a single fused op with no host sync.
            self.k_cache[batch_indices, :, start_pos] = k_new.squeeze(2)
            self.v_cache[batch_indices, :, start_pos] = v_new.squeeze(2)
        else:
            # Mixed batch prefill path: each slot writes its own contiguous
            # range starting at start_pos[b, 0]. The previous implementation
            # used a Python-level ``for`` loop with ``.item()`` to materialize
            # one scalar start position per batch — that stalled the pipeline
            # on every step. Here we keep the whole thing on-device with one
            # advanced-indexed assignment per cache (k, v).
            #
            # We assume start_pos[b] is the contiguous range
            # [s, s+1, ..., s+seq_len_new-1] (i.e. ``position_ids``). The
            # overflow check uses ``start_pos[:, 0]`` (one tensor op, no sync).
            batch_starts = start_pos[:, 0]
            overflow_mask = (batch_starts + seq_len_new) > self.max_seq_len
            if overflow_mask.any():
                overflow_slots = batch_indices[overflow_mask].tolist()
                raise ValueError(
                    f"Cache overflow for slots {overflow_slots} (start_pos + "
                    f"seq_len_new > max_seq_len={self.max_seq_len})"
                )

            b_curr = batch_indices.size(0)
            n_kv = k_new.size(1)
            d_dim = k_new.size(3)

            # Flatten [B, S] -> [B*S]
            b_idx = batch_indices.view(b_curr, 1).expand(b_curr, seq_len_new).reshape(-1)
            s_idx = start_pos.reshape(-1)

            # Permute [B, N_kv, S, D] -> [B, S, N_kv, D] then flatten -> [B*S, N_kv, D]
            # The reshape is a view when memory is contiguous (it is here because
            # permute + reshape is followed by assignment, not by a graph op).
            k_flat = k_new.permute(0, 2, 1, 3).reshape(b_curr * seq_len_new, n_kv, d_dim)
            v_flat = v_new.permute(0, 2, 1, 3).reshape(b_curr * seq_len_new, n_kv, d_dim)

            # Advanced indexing with broadcasting on the head dim. The whole
            # write happens in one scatter-style kernel per cache; no host sync.
            n_kv_idx = torch.arange(n_kv, device=k_new.device)
            self.k_cache[b_idx[:, None], n_kv_idx[None, :], s_idx[:, None]] = k_flat
            self.v_cache[b_idx[:, None], n_kv_idx[None, :], s_idx[:, None]] = v_flat

        # Return the updated cache for these indices.
        # We return the full pre-allocated buffer [B_curr, N_kv, max_seq_len, D]
        # to ensure compatibility with the global attention masks used in continuous batching.
        return self.k_cache[batch_indices], self.v_cache[batch_indices]

    def reset(self) -> None:
        """Reset cache to empty state (does not deallocate memory)."""
        self._seq_len = 0

    def get_usable_length(self, new_tokens: int) -> int:
        """Get the usable cache length after adding new tokens."""
        return self._seq_len + new_tokens

    @classmethod
    def from_model_config(
        cls,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[KVCache]:
        """Create a list of KVCache objects, one per transformer layer.

        Args:
            max_batch_size: Maximum batch size.
            max_seq_len: Maximum sequence length.
            num_layers: Number of transformer layers.
            num_kv_heads: Number of KV heads per layer.
            head_dim: Dimension per head.
            device: Target device.
            dtype: Data type.

        Returns:
            List of KVCache objects, one for each layer.
        """
        return [cls(max_batch_size, max_seq_len, num_kv_heads, head_dim, device, dtype) for _ in range(num_layers)]


def create_decoder_kv_caches(model: Any, batch_size: int) -> list[KVCache]:
    """Create per-layer KV caches sized for a DecoderModel."""
    block = model.transformer_blocks[0]
    num_kv_heads = block.self_attn.num_kv_heads
    head_dim = block.self_attn.head_dim
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    return KVCache.from_model_config(
        max_batch_size=batch_size,
        max_seq_len=model.max_seq_len,
        num_layers=len(model.transformer_blocks),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )


def reset_all_caches(caches: list[KVCache]) -> None:
    """Reset all caches in a list."""
    for cache in caches:
        cache.reset()
