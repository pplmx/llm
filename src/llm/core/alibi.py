"""
ALiBi (Attention with Linear Biases) Module.

Implements ALiBi, a simple position encoding method that adds linear
biases to attention scores based on token distance.

Reference: https://arxiv.org/abs/2108.12409
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    Compute ALiBi slopes for each attention head.

    Slopes form a geometric sequence: 2^(-8/n), 2^(-16/n), ..., 2^(-8)
    where n is the number of heads.

    Args:
        num_heads: Number of attention heads

    Returns:
        Tensor of shape [num_heads] containing slopes
    """
    # Get closest power of 2 for consistent slopes
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))

    # Compute base for the geometric sequence
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))

    # Generate slopes for powers of 2
    powers = torch.arange(1, closest_power_of_2 + 1)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        # If num_heads is not a power of 2, compute extra slopes
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        num_remaining = num_heads - closest_power_of_2
        extra_powers = torch.arange(1, 2 * num_remaining + 1, 2)
        extra_slopes = torch.pow(extra_base, extra_powers)
        slopes = torch.cat([slopes, extra_slopes])

    return slopes


def build_alibi_bias(
    num_heads: int,
    seq_len: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Build the ALiBi bias matrix.

    Args:
        num_heads: Number of attention heads
        seq_len: Sequence length
        device: Target device
        dtype: Target data type

    Returns:
        Tensor of shape [1, num_heads, seq_len, seq_len]
    """
    # Get slopes for each head
    slopes = get_alibi_slopes(num_heads)
    slopes = slopes.to(device=device, dtype=dtype)

    # Create distance matrix: position i attending to position j
    # Distance = j - i (positive for future, negative for past)
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len]

    # Compute bias: -slope * |distance| (penalize far tokens)
    # For causal attention, we only care about past positions (distance <= 0)
    # Use distance directly (negative for past) as bias
    bias = distance.unsqueeze(0) * slopes.unsqueeze(1).unsqueeze(2)  # [num_heads, seq_len, seq_len]

    return bias.unsqueeze(0)  # [1, num_heads, seq_len, seq_len]


class ALiBiPositionBias(nn.Module):
    """
    ALiBi Position Bias Module.

    Generates position-dependent biases to be added to attention scores.

    Args:
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length for cached bias
        device: Target device
        dtype: Target data type
    """

    def __init__(
        self,
        num_heads: int,
        max_seq_len: int = 2048,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Register slopes as buffer (not learned)
        slopes = get_alibi_slopes(num_heads)
        self.register_buffer("slopes", slopes, persistent=False)

        # Cache for bias matrix
        self._cached_bias: torch.Tensor | None = None
        self._cached_seq_len: int = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """Update cached bias if sequence length changed."""
        if seq_len > self._cached_seq_len or self._cached_bias is None:
            self._cached_seq_len = max(seq_len, self.max_seq_len)
            self._cached_bias = build_alibi_bias(
                self.num_heads,
                self._cached_seq_len,
                device=device,
                dtype=dtype,
            )

    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        Add ALiBi bias to attention scores.

        Args:
            attention_scores: Attention scores of shape [batch, heads, seq_len, seq_len]

        Returns:
            Attention scores with ALiBi bias added
        """
        seq_len = attention_scores.size(-1)
        self._update_cache(seq_len, attention_scores.device, attention_scores.dtype)

        # Extract relevant portion of cached bias
        bias = self._cached_bias[:, :, :seq_len, :seq_len]
        return attention_scores + bias

    def get_bias(
        self,
        seq_len: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Get ALiBi bias matrix for given sequence length.

        Args:
            seq_len: Sequence length
            device: Target device
            dtype: Target data type

        Returns:
            Bias tensor of shape [1, num_heads, seq_len, seq_len]
        """
        device = device or self.slopes.device
        dtype = dtype or self.slopes.dtype
        self._update_cache(seq_len, device, dtype)
        return self._cached_bias[:, :, :seq_len, :seq_len]

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, max_seq_len={self.max_seq_len}"
