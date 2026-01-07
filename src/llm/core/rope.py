"""
Rotary Position Embedding (RoPE) Module.

Implements RoPE with scaling support for extended context lengths.
Supports linear scaling, dynamic scaling, and NTK-aware scaling.

Reference: https://arxiv.org/abs/2104.09864
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Encodes position information by rotating query and key vectors.

    Args:
        dim: Dimension of the embedding (typically head_dim)
        max_seq_len: Maximum sequence length for precomputed embeddings
        base: Base for computing rotation frequencies (default: 10000)
        scaling_type: Type of RoPE scaling: None, 'linear', 'dynamic', 'ntk'
        scaling_factor: Scaling factor for extended context (default: 1.0)
        device: Device for the embedding
        dtype: Data type for the embedding
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        scaling_type: str | None = None,
        scaling_factor: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor

        # Compute inverse frequencies
        inv_freq = self._compute_inv_freq(device, dtype)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin for efficiency
        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

    def _compute_inv_freq(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Compute inverse frequencies with optional NTK scaling."""
        calc_dtype = dtype if dtype is not None and dtype.is_floating_point else torch.float32

        if self.scaling_type == "ntk" and self.scaling_factor > 1.0:
            # NTK-aware scaling: adjust base instead of positions
            base = self.base * (
                (self.scaling_factor * self.max_seq_len / self.max_seq_len) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
        else:
            base = self.base

        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=device, dtype=calc_dtype) / self.dim))
        return inv_freq

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """Update cached cos/sin values if sequence length changed."""
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = max(seq_len, self.max_seq_len)

            # Position indices
            t = torch.arange(self._seq_len_cached, device=device, dtype=self.inv_freq.dtype)

            # Apply linear or dynamic scaling to positions
            if self.scaling_type == "linear" and self.scaling_factor > 1.0:
                t = t / self.scaling_factor
            elif self.scaling_type == "dynamic" and seq_len > self.max_seq_len:
                # Dynamic scaling: scale positions based on current sequence length
                scale = seq_len / self.max_seq_len
                t = t / scale

            # Compute frequencies: [seq_len, dim/2]
            freqs = torch.outer(t, self.inv_freq)

            # Compute cos and sin: [seq_len, dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to query and key tensors.

        Args:
            q: Query tensor of shape [batch, heads, seq_len, head_dim]
            k: Key tensor of shape [batch, heads, seq_len, head_dim]
            position_ids: Optional position indices [batch, seq_len]

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        seq_len = q.size(2)
        self._update_cos_sin_cache(seq_len, q.device, q.dtype)

        if position_ids is None:
            cos = self._cos_cached[:seq_len]
            sin = self._sin_cached[:seq_len]
        else:
            cos = self._cos_cached[position_ids]
            sin = self._sin_cached[position_ids]

        # Reshape for broadcasting: [1, 1, seq_len, dim] or [batch, 1, seq_len, dim]
        if position_ids is None:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_embed = apply_rotary_pos_emb(q, cos, sin)
        k_embed = apply_rotary_pos_emb(k, cos, sin)

        return q_embed, k_embed

    def extra_repr(self) -> str:
        scaling_info = (
            f", scaling_type={self.scaling_type}, scaling_factor={self.scaling_factor}" if self.scaling_type else ""
        )
        return f"dim={self.dim}, max_seq_len={self.max_seq_len}, base={self.base}{scaling_info}"


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embedding to a tensor.

    Args:
        x: Input tensor of shape [..., seq_len, dim]
        cos: Cosine embeddings
        sin: Sine embeddings

    Returns:
        Rotated tensor of the same shape
    """
    return (x * cos) + (rotate_half(x) * sin)


def get_rope_scaling_factor(
    seq_len: int,
    max_trained_len: int,
    scaling_type: str = "linear",
) -> float:
    """
    Compute appropriate RoPE scaling factor.

    Args:
        seq_len: Current sequence length
        max_trained_len: Maximum length the model was trained on
        scaling_type: Type of scaling ('linear' or 'dynamic')

    Returns:
        Scaling factor
    """
    if seq_len <= max_trained_len:
        return 1.0

    if scaling_type == "linear":
        return seq_len / max_trained_len
    elif scaling_type == "dynamic":
        return math.sqrt(seq_len / max_trained_len)
    else:
        return 1.0
