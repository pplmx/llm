import math

import torch
from torch import Tensor, nn

from .dot_product_attn import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism using a custom attention function.

    Supports Pre-LN and Post-LN normalization with improved performance.

    Args:
        hidden_size: Total model dimension.
        num_heads: Number of attention heads. Must divide hidden_size.
        p: Dropout probability for the attention weights (inside attention calc). Defaults to 0.1.
        bias: Whether to use bias in linear layers. Defaults to True.
        eps: Epsilon for Layer Normalization. Defaults to 1e-5.
        norm_first: Use Pre-LN (True) or Post-LN (False). Defaults to True.
        is_causal: Use causal masking in attention. Defaults to False.
        device: Target device. Defaults to None (inferred).
        dtype: Target data type. Defaults to None (inferred).

    Attributes:
        head_dim: Dimension of each head.
        scale: Pre-calculated attention scaling factor (sqrt(head_dim)).
        qkv_proj: Combined QKV projection layer.
        out_proj: Output projection layer.
        norm: Layer normalization module.
        dropout: Dropout layer applied to the output projection.
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

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.norm_first = norm_first
        self.is_causal = is_causal
        self.p = p

        # --- Layers ---
        self.norm = nn.LayerNorm(hidden_size, eps=eps, **factory_kwargs)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(p)

        self._init_weights()

    def _init_weights(self):
        """Initialize linear layer weights."""
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            hidden_states: Input tensor [B, S, H].
            attn_mask: Optional mask passed to the attention function.
                       Expected format depends on the function (e.g., bool `True`=mask).

        Returns:
            Output tensor [B, S, H].
        """
        batch_size, seq_len, _ = hidden_states.size()
        residual = hidden_states

        # 1. Layer Normalization (Pre-LN)
        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        # 2. Project Q, K, V
        qkv = (
            self.qkv_proj(hidden_states)  # [B, S, 3*H]
            .reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)  # [B, S, 3, N, D]
            .permute(2, 0, 3, 1, 4)  # [3, B, N, S, D]
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each [B, H, S, D]

        # 3. Adapt attn mask to [B, N, S, S]
        if attn_mask is not None and attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)

        # 4. Scaled Dot-Product Attention
        attn_output = scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.p if self.training else 0.0,  # Apply dropout only during training
            is_causal=self.is_causal,
            scale=self.scale,
        )

        # 5. Combine Heads: [B, N, S, D] -> [B, S, H]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)

        # 6. Output Projection & Dropout
        output = self.dropout(self.out_proj(attn_output))

        # 7. Residual Connection
        output = output + residual  # Slightly more efficient than residual + output

        # 8. Layer Normalization (Post-LN)
        if not self.norm_first:
            output = self.norm(output)

        return output
