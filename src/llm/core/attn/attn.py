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

        assert hidden_size % num_heads == 0, "Hidden size must be divisible by number of attention heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.norm_first = norm_first
        self.num_latents = num_latents
        self.latent_dim = latent_dim if latent_dim is not None else hidden_size
        self.is_causal = is_causal
        self.p = p

        # Layer Normalization (applied first in forward pass if norm_first is True)
        if norm_first:
            self.norm = nn.LayerNorm(hidden_size, eps=eps, device=device, dtype=dtype)

        # Learnable latent vectors
        self.latents = nn.Parameter(torch.randn(num_latents, self.latent_dim, device=device, dtype=dtype))

        # Latent projections
        self.latent_q_proj = nn.Linear(self.latent_dim, hidden_size, bias=bias, device=device, dtype=dtype)
        self.latent_v_proj = nn.Linear(self.latent_dim, hidden_size, bias=bias, device=device, dtype=dtype)

        # Combined Input projections for Key and Value
        self.input_kv_proj = nn.Linear(hidden_size, 2 * hidden_size, bias=bias, device=device, dtype=dtype)

        # Output projections
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)

        # Dropout layers
        self.dropout = nn.Dropout(p)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes the weights of the multi-latent attention layers."""
        # Initialize latent vectors with a normal distribution
        nn.init.normal_(self.latents, std=0.02)

        # Initialize linear projections
        nn.init.xavier_uniform_(self.latent_q_proj.weight)
        nn.init.xavier_uniform_(self.latent_v_proj.weight)
        nn.init.xavier_uniform_(self.input_kv_proj.weight)  # Initialize combined KV projection
        nn.init.xavier_uniform_(self.out_proj.weight)

        # Initialize biases to zero
        if self.latent_q_proj.bias is not None:
            nn.init.zeros_(self.latent_q_proj.bias)
        if self.latent_v_proj.bias is not None:
            nn.init.zeros_(self.latent_v_proj.bias)
        if self.input_kv_proj.bias is not None:  # Initialize combined KV bias
            nn.init.zeros_(self.input_kv_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _reshape_for_multihead_attention(self, x: Tensor, from_latents: bool = False) -> Tensor:
        """
        Reshapes the input tensor for multi-head attention computation.

        Args:
            x: Input tensor with shape [batch_size, seq_len, hidden_size] or
               [num_latents, hidden_size] if from_latents is True.
            from_latents: Whether the input is from latent vectors.

        Returns:
            Reshaped tensor with shape:
            - [batch_size, seq_len, num_heads, head_dim] and transposed if from_latents is False
            - [batch_size, num_latents, num_heads, head_dim] and transposed if from_latents is True
        """
        if from_latents:
            # For latent vectors: [num_latents, hidden_size]
            x = x.view(self.num_latents, self.num_heads, self.head_dim)
            # Unsqueeze batch dimension: [1, num_latents, num_heads, head_dim]
            return x.unsqueeze(0).transpose(1, 2)
        else:
            # For input sequences: [batch_size, seq_len, hidden_size]
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

        # Apply Layer Normalization if enabled (Pre-LN architecture)
        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        batch_size, seq_len, _ = hidden_states.size()

        # Phase 1: Latents attend to input sequence
        # Project inputs to keys and values using the combined layer
        kv = self.input_kv_proj(hidden_states)  # [batch_size, seq_len, 2 * hidden_size]
        # Split the combined tensor into input_keys and input_values
        input_keys, input_values = kv.chunk(2, dim=-1)  # Two tensors of [batch_size, seq_len, hidden_size]

        # Project latents to queries
        latent_queries = self.latent_q_proj(self.latents)  # [num_latents, hidden_size]

        # Reshape for multi-head attention
        input_keys = self._reshape_for_multihead_attention(input_keys)  # [batch_size, num_heads, seq_len, head_dim]
        input_values = self._reshape_for_multihead_attention(input_values)  # [batch_size, num_heads, seq_len, head_dim]
        latent_queries = self._reshape_for_multihead_attention(latent_queries, from_latents=True)
        # [batch_size, num_heads, num_latents, head_dim]

        # Expand attention mask for latent dimension if provided
        expanded_mask = None
        if attn_mask is not None:
            # Expand mask to cover the latent dimension
            # [batch_size, 1, 1, seq_len] -> [batch_size, 1, num_latents, seq_len]
            expanded_mask = attn_mask.expand(-1, -1, self.num_latents, -1)

        # Compute scaled dot-product attention using the custom function
        latent_output = scaled_dot_product_attention(
            query=latent_queries,
            key=input_keys,
            value=input_values,
            attn_mask=expanded_mask,
            dropout_p=self.p,
            is_causal=self.is_causal,
            scale=self.scale,
        )  # [batch_size, num_heads, num_latents, head_dim]

        # Phase 2: Transform updated latents
        # Reshape latent context: [batch_size, num_heads, num_latents, head_dim] -> [batch_size, num_latents, hidden_size]
        latent_output = latent_output.transpose(1, 2).contiguous()
        latent_output = latent_output.view(batch_size, self.num_latents, self.hidden_size)

        # Project latents to output values
        latent_output_values = self.latent_v_proj(latent_output)  # [batch_size, num_latents, hidden_size]

        # Reshape for final projection
        # Reshape to match the input sequence: [batch_size, seq_len, hidden_size]
        # We use a simple broadcasted expansion for simplicity
        # This effectively copies the same latent outputs to each position in the sequence
        output = latent_output_values.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)

        # Project to final representation
        output = self.out_proj(output)
        output = self.dropout(output)

        # Add residual connection
        output = residual + output

        return output
