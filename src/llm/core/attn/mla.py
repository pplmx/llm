import math

import torch
from torch import Tensor, nn

from .sdpa import sdpa


class MultiLatentAttention(nn.Module):
    """
    Multi-Latent Attention mechanism implementation.

    Similar to Multi-Head Attention but incorporates a set of learnable latent vectors
    that can capture different aspects of the input and enhance attention computations.

    Args:
        hidden_size: Hidden dimension size.
        num_heads: Number of attention heads. Defaults to 8.
        num_latents: Number of latent vectors. Defaults to 16.
        latent_dim: Size of each latent vector. If None, equals hidden_size. Defaults to None.
        dropout_p: Dropout probability. Defaults to 0.1.
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
        num_latents: int = 16,
        latent_dim: int | None = None,
        dropout_p: float = 0.1,
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
        self.scale = 1 / math.sqrt(self.head_dim)
        self.norm_first = norm_first
        self.num_latents = num_latents
        self.latent_dim = latent_dim if latent_dim is not None else hidden_size
        self.is_causal = is_causal
        self.dropout_p = dropout_p

        # Layer Normalization - shared for all attention operations
        self.norm = nn.LayerNorm(hidden_size, eps=eps, **factory_kwargs)

        # Learnable latent vectors - initialized directly with normal distribution
        self.latents = nn.Parameter(torch.randn(1, num_latents, self.latent_dim, **factory_kwargs) * 0.02)

        # Latent projections
        self.latent_q_proj = nn.Linear(self.latent_dim, hidden_size, bias=bias, **factory_kwargs)
        self.latent_v_proj = nn.Linear(hidden_size, self.latent_dim, bias=bias, **factory_kwargs)
        self.latent_output_proj = nn.Linear(self.latent_dim, hidden_size, bias=bias, **factory_kwargs)

        # Input projection for Key and Value
        self.input_kv_proj = nn.Linear(hidden_size, 2 * hidden_size, bias=bias, **factory_kwargs)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, **factory_kwargs)

        # Single dropout layer
        self.dropout = nn.Dropout(dropout_p)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize module parameters with optimized scheme."""
        # Xavier/Glorot uniform initialization for all linear layers
        for module in [
            self.latent_q_proj,
            self.latent_v_proj,
            self.latent_output_proj,
            self.input_kv_proj,
            self.out_proj,
        ]:
            # Use improved initialization with gain based on activation
            gain = 1.0  # Linear activation gain
            fan_in, fan_out = module.weight.shape
            std = gain * math.sqrt(2.0 / (fan_in + fan_out))
            nn.init.trunc_normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _latent_attention(self, k: Tensor, v: Tensor, batch_size: int, attn_mask: Tensor | None = None) -> Tensor:
        """
        Process latent attention computation as a separate method for clarity.
        This function computes the attention between latent queries and input sequence.

        Args:
            k: Key tensor with shape [batch_size, num_heads, seq_len, head_dim]
            v: Value tensor with shape [batch_size, num_heads, seq_len, head_dim]
            batch_size: Batch size
            attn_mask: Optional attention mask

        Returns:
            Processed latent output with shape [batch_size, num_latents, hidden_size]
        """
        # Expand and project latent queries in one operation
        latent_q = self.latents.expand(batch_size, -1, -1)
        latent_q = self.latent_q_proj(latent_q.reshape(batch_size * self.num_latents, self.latent_dim))
        latent_q = latent_q.view(batch_size, self.num_latents, self.num_heads, self.head_dim)
        latent_q = latent_q.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_latents, head_dim]

        # Process attention mask for latent queries if provided
        if attn_mask is not None:
            attn_mask = attn_mask.expand(-1, -1, self.num_latents, -1)

        # Compute attention with conditional dropout during training
        latent_output = sdpa(
            query=latent_q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=self.is_causal,
            scale=self.scale,
        )  # [batch_size, num_heads, num_latents, head_dim]

        # Reshape for further processing
        latent_output = latent_output.permute(0, 2, 1, 3).reshape(batch_size, self.num_latents, self.hidden_size)

        # Transform through latent dimension
        latent_output = self.latent_v_proj(latent_output)
        latent_output = self.latent_output_proj(latent_output.reshape(batch_size * self.num_latents, self.latent_dim))

        return latent_output.reshape(batch_size, self.num_latents, self.hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Optimized forward pass for the multi-latent attention mechanism.

        Args:
            hidden_states: Input tensor with shape [batch_size, seq_len, hidden_size].
            attn_mask: Optional mask tensor with shape [batch_size, 1, 1, seq_len].
                       1 indicates positions to attend to, 0 indicates positions to mask.

        Returns:
            Output tensor with shape [batch_size, seq_len, hidden_size].
        """
        # Store residual connection
        residual = hidden_states

        # Get shape parameters once
        batch_size, seq_len = hidden_states.shape[:2]

        # Apply Layer Normalization if Pre-LN architecture
        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        # Project input to key-value pairs in a single operation
        kv_proj = self.input_kv_proj(hidden_states)
        kv_proj = kv_proj.view(batch_size, seq_len, 2, self.num_heads, self.head_dim)
        kv_proj = kv_proj.permute(2, 0, 3, 1, 4)
        k, v = kv_proj[0], kv_proj[1]  # [batch_size, num_heads, seq_len, head_dim]

        # Process latent attention
        latent_output = self._latent_attention(k, v, batch_size, attn_mask)

        # Compute uniform weights across latents by default
        latent_weights = torch.ones(batch_size, 1, self.num_latents, device=latent_output.device) / self.num_latents

        # Apply weights to latent outputs
        output = torch.bmm(latent_weights, latent_output)

        # Expand to sequence length dimension efficiently
        output = output.expand(-1, seq_len, -1)

        # Apply output projection and dropout
        output = self.out_proj(output)
        output = self.dropout(output)

        # Add residual connection
        output = output + residual

        # Apply Layer Normalization if Post-LN architecture
        if not self.norm_first:
            output = self.norm(output)

        return output
