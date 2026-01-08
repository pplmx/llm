"""
KV Cache manager for efficient autoregressive generation.

This module provides a pre-allocated cache that avoids repeated memory allocations
during token generation, significantly improving inference performance.
"""

from __future__ import annotations

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


def reset_all_caches(caches: list[KVCache]) -> None:
    """Reset all caches in a list."""
    for cache in caches:
        cache.reset()
