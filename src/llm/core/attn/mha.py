import math

import torch
from torch import Tensor, nn

from .dot_product_attn import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism implementation as described in 'Attention Is All You Need'.

    Performs multiple attention operations in parallel and concatenates the results.

    Args:
        hidden_size: Hidden dimension size.
        num_heads: Number of attention heads. Defaults to 8.
        p: Dropout probability. Defaults to 0.1.
        bias: Whether to use bias in the linear layers. Defaults to True.
        eps: Epsilon value for Layer Normalization. Defaults to 1e-5.
        norm_first: Whether to use Layer Normalization before attention. Defaults to True.
        is_causal: Whether to use causal attention. Defaults to False.
        device: Device for the model.
        dtype: Data type for the model parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        p: float = 0.1,
        bias: bool = True,
        eps: float = 1e-5,
        norm_first: bool = True,
        is_causal: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        assert hidden_size % num_heads == 0, "Hidden size must be divisible by number of attention heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.norm_first = norm_first
        self.is_causal = is_causal
        self.p = p

        # Layer Normalization (applied first in forward pass if norm_first is True)
        if norm_first:
            self.norm = nn.LayerNorm(hidden_size, eps=eps, device=device, dtype=dtype)

        # Combined Linear projections for Query, Key, and Value
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias, device=device, dtype=dtype)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)

        # Dropout layers
        self.dropout = nn.Dropout(p)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes the weights of the multi-head attention layers."""
        # Initialize combined input projection
        # Splitting the weight matrix for potentially different initializations if needed,
        # but Xavier uniform for the whole matrix is common.
        nn.init.xavier_uniform_(self.in_proj.weight)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)

        # Initialize output projection
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _reshape_for_multihead_attention(self, x: Tensor) -> Tensor:
        """
        Reshapes the input tensor for multi-head attention computation.

        Args:
            x: Input tensor with shape [batch_size, seq_len, hidden_size].

        Returns:
            Reshaped tensor with shape [batch_size, seq_len, num_heads, head_dim].
            Then transposed to [batch_size, num_heads, seq_len, head_dim].
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        return x.transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass for the multi-head attention mechanism.

        Args:
            hidden_states: Input tensor, shape [batch_size, seq_len, hidden_size].
            attn_mask: Optional mask tensor, shape [batch_size, 1, 1, seq_len] or
                        [batch_size, 1, seq_len, seq_len]. 1 indicates positions to attend to,
                        0 indicates positions to mask out.

        Returns:
            Output tensor with shape [batch_size, seq_len, hidden_size].
        """
        # Save residual for later
        residual = hidden_states

        # Apply Layer Normalization if enabled (Pre-LN architecture)
        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        batch_size, seq_len, _ = hidden_states.size()

        # Project inputs to queries, keys, and values using the combined layer
        qkv = self.in_proj(hidden_states)  # [batch_size, seq_len, 3 * hidden_size]

        # Split the combined tensor into query, key, and value
        # qkv: [batch_size, seq_len, 3 * hidden_size] -> three tensors of [batch_size, seq_len, hidden_size]
        query, key, value = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        # [batch_size, num_heads, seq_len, head_dim]
        query = self._reshape_for_multihead_attention(query)
        key = self._reshape_for_multihead_attention(key)
        value = self._reshape_for_multihead_attention(value)

        # Compute scaled dot-product attention using the custom function
        attn_output = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=self.p,
            is_causal=self.is_causal,
            scale=self.scale,
        )  # [batch_size, num_heads, seq_len, head_dim]

        # Reshape back to original dimensions
        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Combine all heads
        # [batch_size, seq_len, num_heads * head_dim] = [batch_size, seq_len, hidden_size]
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # Project the context layer
        output = self.out_proj(attn_output)
        output = self.dropout(output)

        # Add residual connection
        output = residual + output

        return output
