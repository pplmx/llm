import math

import torch
from torch import Tensor, nn

from .dot_product_attn import scaled_dot_product_attention


class Attention(nn.Module):
    """
    Basic Attention mechanism implementation.

    Computes scaled dot-product attention as described in 'Attention Is All You Need'.

    Args:
        hidden_size: Hidden dimension size.
        p: Dropout probability. Defaults to 0.1.
        bias: Whether to use bias in the linear layers. Defaults to True.
        is_causal: Whether to use causal attention. Defaults to False.
        device: Device for the model.
        dtype: Data type for the model parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        p: float = 0.1,
        bias: bool = True,
        is_causal: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.p = p
        self.scale = math.sqrt(hidden_size)
        self.is_causal = is_causal

        # Linear projections for Query, Key, and Value
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes the weights of the attention layers."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass for the basic attention mechanism.

        Args:
            hidden_states: Input tensor, shape [batch_size, seq_len, hidden_size].
            attn_mask: Optional mask tensor, shape [batch_size, 1, 1, seq_len] or
                        [batch_size, 1, seq_len, seq_len]. 1 indicates positions to attend to,
                        0 indicates positions to mask out.

        Returns:
            Output tensor with shape [batch_size, seq_len, hidden_size].
        """
        # Project inputs to queries, keys, and values
        query = self.q_proj(hidden_states)  # [batch_size, seq_len, hidden_size]
        key = self.k_proj(hidden_states)  # [batch_size, seq_len, hidden_size]
        value = self.v_proj(hidden_states)  # [batch_size, seq_len, hidden_size]

        # Add head dim(Single Head) - (batch_size, 1, seq_len, hidden_size)
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        # Extend mask dim to broadcast
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)

        # Compute Attention
        attn_output = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=self.p,
            is_causal=self.is_causal,
            scale=self.scale,
        )  # (batch_size, 1, seq_len, hidden_size)

        # Remove head dim
        attn_output = attn_output.squeeze(1)  # (batch_size, seq_len, hidden_size)

        # Project the context layer
        output = self.out_proj(attn_output)  # (batch_size, seq_len, hidden_size)

        return output
