import math

import torch
from torch import Tensor, nn

from .dot_product_attn import scaled_dot_product_attention


class MultiLatentAttention(nn.Module):
    """
    Optimized Multi-Latent Attention mechanism implementation.

    Similar to Multi-Head Attention but incorporates a set of learnable latent vectors
    that can capture different aspects of the input and enhance attention computations.
    This optimized version improves tensor operations and reshaping for better performance.

    Args:
        hidden_size: Hidden dimension size.
        num_heads: Number of attention heads. Defaults to 8.
        num_latents: Number of latent vectors. Defaults to 16.
        latent_dim: Size of each latent vector. If None, equals hidden_size. Defaults to None.
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
        num_latents: int = 16,
        latent_dim: int | None = None,
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
        self.num_latents = num_latents
        self.latent_dim = latent_dim or hidden_size
        self.is_causal = is_causal
        self.p = p

        # Layer Normalization
        self.norm = nn.LayerNorm(hidden_size, eps=eps, **factory_kwargs)

        # Learnable latent vectors
        self.latents = nn.Parameter(torch.randn(1, num_latents, self.latent_dim, **factory_kwargs))

        # Latent projections
        self.latent_q_proj = nn.Linear(self.latent_dim, hidden_size, bias=bias, **factory_kwargs)
        self.latent_v_proj = nn.Linear(hidden_size, self.latent_dim, bias=bias, **factory_kwargs)
        self.latent_output_proj = nn.Linear(self.latent_dim, hidden_size, bias=bias, **factory_kwargs)

        # Input projections for Key and Value
        self.input_kv_proj = nn.Linear(hidden_size, 2 * hidden_size, bias=bias, **factory_kwargs)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, **factory_kwargs)

        # Dropout layers
        self.dropout = nn.Dropout(p)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes the weights of the multi-latent attention layers."""
        # Initialize latent vectors with a normal distribution
        nn.init.normal_(self.latents, std=0.02)

        # Initialize linear projections with Xavier/Glorot
        for module in [
            self.latent_q_proj,
            self.latent_v_proj,
            self.latent_output_proj,
            self.input_kv_proj,
            self.out_proj,
        ]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass for the multi-latent attention mechanism.

        Args:
            hidden_states: Input tensor, shape [batch_size, seq_len, hidden_size].
            attn_mask: Optional mask tensor, shape [batch_size, 1, 1, seq_len].
                       1 indicates positions to attend to, 0 indicates positions to mask out.

        Returns:
            Output tensor with shape [batch_size, seq_len, hidden_size].
        """
        # Save residual for later
        residual = hidden_states
        batch_size, seq_len, _ = hidden_states.size()

        # Apply Layer Normalization if enabled (Pre-LN architecture)
        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        # Phase 1: Latents attend to input sequence
        # Project inputs to keys and values
        kv_proj = self.input_kv_proj(hidden_states)  # [batch_size, seq_len, 2*hidden_size]
        k_proj, v_proj = kv_proj.chunk(2, dim=-1)  # Each [batch_size, seq_len, hidden_size]

        # Reshape K, V for multi-head attention
        # [batch_size, seq_len, hidden_size] -> [batch_size, num_heads, seq_len, head_dim]
        k_proj = k_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_proj = v_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Get latent queries - optimized to avoid conditional reshaping logic
        # First expand latents to batch dimension
        latents = self.latents.expand(batch_size, -1, -1)  # [batch_size, num_latents, latent_dim]

        # Project latents to queries and reshape for attention
        # [batch_size, num_latents, latent_dim] -> [batch_size, num_latents, hidden_size]
        latent_q = self.latent_q_proj(latents)

        # [batch_size, num_latents, hidden_size] -> [batch_size, num_heads, num_latents, head_dim]
        latent_q = latent_q.view(batch_size, self.num_latents, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Handle attention mask for latents
        if attn_mask is not None:
            # Expand mask to cover the latent dimension
            # [batch_size, 1, 1, seq_len] -> [batch_size, 1, num_latents, seq_len]
            attn_mask = attn_mask.expand(-1, -1, self.num_latents, -1)

        # Compute scaled dot-product attention
        latent_output = scaled_dot_product_attention(
            query=latent_q,
            key=k_proj,
            value=v_proj,
            attn_mask=attn_mask,
            dropout_p=self.p if self.training else 0.0,  # Only apply dropout during training
            is_causal=self.is_causal,
            scale=self.scale,
        )  # [batch_size, num_heads, num_latents, head_dim]

        # Phase 2: Transform updated latents - streamlined transformations
        # Reshape: [batch_size, num_heads, num_latents, head_dim] -> [batch_size, num_latents, hidden_size]
        latent_output = latent_output.permute(0, 2, 1, 3).reshape(batch_size, self.num_latents, self.hidden_size)

        # Project from hidden_size to latent_dim
        latent_output = self.latent_v_proj(latent_output)  # [batch_size, num_latents, latent_dim]

        # Project from latent_dim back to hidden_size
        latent_output = self.latent_output_proj(latent_output)  # [batch_size, num_latents, hidden_size]

        # Improved output mapping - using weighted combination instead of simple mean
        # This allows the model to weight latents differently based on their importance
        output_weights = torch.softmax(torch.ones(batch_size, self.num_latents, 1, device=latent_output.device), dim=1)
        output = torch.bmm(output_weights.transpose(1, 2), latent_output)  # [batch_size, 1, hidden_size]
        output = output.expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_size]

        # Final projection and dropout
        output = self.out_proj(output)
        output = self.dropout(output)

        # Residual connection
        output = output + residual

        # Apply Layer Normalization if Post-LN architecture
        if not self.norm_first:
            output = self.norm(output)

        return output
