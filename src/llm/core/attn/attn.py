import math

import torch
from torch import Tensor, nn

from llm.core.attn import scaled_dot_product_attention


class Attention(nn.Module):
    """
    Basic Attention mechanism implementation.

    Computes scaled dot-product attention as described in 'Attention Is All You Need'.

    Args:
        hidden_size: Hidden dimension size.
        dropout_p: Dropout probability. Defaults to 0.1.
        bias: Whether to use bias in the linear layers. Defaults to True.
        device: Device for the model.
        dtype: Data type for the model parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout_p: float = 0.1,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.scaling = math.sqrt(hidden_size)

        # Linear projections for Query, Key, and Value
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes the weights of the attention layers."""
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.query_proj.bias is not None:
            nn.init.zeros_(self.query_proj.bias)
        if self.key_proj.bias is not None:
            nn.init.zeros_(self.key_proj.bias)
        if self.value_proj.bias is not None:
            nn.init.zeros_(self.value_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass for the basic attention mechanism.

        Args:
            hidden_states: Input tensor, shape [batch_size, seq_len, hidden_size].
            attention_mask: Optional mask tensor, shape [batch_size, 1, 1, seq_len] or
                           [batch_size, 1, seq_len, seq_len]. 1 indicates positions to attend to,
                           0 indicates positions to mask out.

        Returns:
            Output tensor with shape [batch_size, seq_len, hidden_size].
        """
        # Project inputs to queries, keys, and values
        query = self.query_proj(hidden_states)  # [batch_size, seq_len, hidden_size]
        key = self.key_proj(hidden_states)  # [batch_size, seq_len, hidden_size]
        value = self.value_proj(hidden_states)  # [batch_size, seq_len, hidden_size]

        # Add head dim(Single Head) - (batch_size, 1, seq_len, hidden_size)
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        # Extend mask dim to broadcast
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)

        # Compute Attention
        context_layer = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attention_mask,
            dropout_p=self.dropout_p,
            is_causal=self.is_causal,
            scale=self.scaling,
        )  # (batch_size, 1, seq_len, hidden_size)

        # Remove head dim
        context_layer = context_layer.squeeze(1)  # (batch_size, seq_len, hidden_size)

        # Project the context layer
        output = self.out_proj(context_layer)  # (batch_size, seq_len, hidden_size)

        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism implementation as described in 'Attention Is All You Need'.

    Performs multiple attention operations in parallel and concatenates the results.

    Args:
        hidden_size: Hidden dimension size.
        num_attention_heads: Number of attention heads. Defaults to 8.
        dropout_p: Dropout probability. Defaults to 0.1.
        bias: Whether to use bias in the linear layers. Defaults to True.
        layer_norm_eps: Epsilon value for Layer Normalization. Defaults to 1e-5.
        use_layer_norm: Whether to use Layer Normalization. Defaults to True.
        device: Device for the model.
        dtype: Data type for the model parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        dropout_p: float = 0.1,
        bias: bool = True,
        layer_norm_eps: float = 1e-5,
        use_layer_norm: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        assert hidden_size % num_attention_heads == 0, "Hidden size must be divisible by number of attention heads"

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.scaling = math.sqrt(self.head_size)
        self.use_layer_norm = use_layer_norm

        # Layer Normalization (applied first in forward pass if use_layer_norm is True)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, device=device, dtype=dtype)

        # Linear projections for Query, Key, and Value
        self.query = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
        self.key = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
        self.value = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)

        # Dropout layers
        self.attention_dropout = nn.Dropout(dropout_p)
        self.output_dropout = nn.Dropout(dropout_p)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes the weights of the multi-head attention layers."""
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.query.bias is not None:
            nn.init.zeros_(self.query.bias)
        if self.key.bias is not None:
            nn.init.zeros_(self.key.bias)
        if self.value.bias is not None:
            nn.init.zeros_(self.value.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _reshape_for_multihead_attention(self, x: Tensor) -> Tensor:
        """
        Reshapes the input tensor for multi-head attention computation.

        Args:
            x: Input tensor with shape [batch_size, seq_len, hidden_size].

        Returns:
            Reshaped tensor with shape [batch_size, seq_len, num_heads, head_size].
            Then transposed to [batch_size, num_heads, seq_len, head_size].
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.head_size)
        # Transpose to [batch_size, num_heads, seq_len, head_size]
        return x.transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass for the multi-head attention mechanism.

        Args:
            hidden_states: Input tensor, shape [batch_size, seq_len, hidden_size].
            attention_mask: Optional mask tensor, shape [batch_size, 1, 1, seq_len] or
                           [batch_size, 1, seq_len, seq_len]. 1 indicates positions to attend to,
                           0 indicates positions to mask out.

        Returns:
            Output tensor with shape [batch_size, seq_len, hidden_size].
        """
        # Save residual for later
        residual = hidden_states

        # Apply Layer Normalization if enabled (Pre-LN architecture)
        if self.use_layer_norm:
            hidden_states = self.layer_norm(hidden_states)

        batch_size, seq_len, _ = hidden_states.size()

        # Project inputs to queries, keys, and values
        query_states = self.query(hidden_states)  # [batch_size, seq_len, hidden_size]
        key_states = self.key(hidden_states)  # [batch_size, seq_len, hidden_size]
        value_states = self.value(hidden_states)  # [batch_size, seq_len, hidden_size]

        # Reshape for multi-head attention
        # [batch_size, num_heads, seq_len, head_size]
        query_states = self._reshape_for_multihead_attention(query_states)
        key_states = self._reshape_for_multihead_attention(key_states)
        value_states = self._reshape_for_multihead_attention(value_states)

        # Compute scaled dot-product attention
        # [batch_size, num_heads, seq_len, head_size] x [batch_size, num_heads, head_size, seq_len]
        # -> [batch_size, num_heads, seq_len, seq_len]
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / self.scaling

        # Apply attention mask if provided
        if attention_mask is not None:
            # Add large negative value to masked positions to make their softmax output close to 0
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        # Apply dropout to the attention probabilities
        attention_probs = self.attention_dropout(attention_probs)

        # Compute the weighted sum of values
        # [batch_size, num_heads, seq_len, seq_len] x [batch_size, num_heads, seq_len, head_size]
        # -> [batch_size, num_heads, seq_len, head_size]
        context_layer = torch.matmul(attention_probs, value_states)

        # Reshape back to original dimensions
        # [batch_size, num_heads, seq_len, head_size] -> [batch_size, seq_len, num_heads, head_size]
        context_layer = context_layer.transpose(1, 2).contiguous()

        # Combine all heads
        # [batch_size, seq_len, num_heads * head_size] = [batch_size, seq_len, hidden_size]
        context_layer = context_layer.view(batch_size, seq_len, self.hidden_size)

        # Project the context layer
        output = self.out_proj(context_layer)
        output = self.output_dropout(output)

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
        num_attention_heads: Number of attention heads. Defaults to 8.
        num_latents: Number of latent vectors. Defaults to 16.
        latent_size: Size of each latent vector. If None, equals hidden_size. Defaults to None.
        dropout_p: Dropout probability. Defaults to 0.1.
        bias: Whether to use bias in the linear layers. Defaults to True.
        layer_norm_eps: Epsilon value for Layer Normalization. Defaults to 1e-5.
        use_layer_norm: Whether to use Layer Normalization. Defaults to True.
        device: Device for the model.
        dtype: Data type for the model parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        num_latents: int = 16,
        latent_size: int | None = None,
        dropout_p: float = 0.1,
        bias: bool = True,
        layer_norm_eps: float = 1e-5,
        use_layer_norm: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        assert hidden_size % num_attention_heads == 0, "Hidden size must be divisible by number of attention heads"

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.scaling = math.sqrt(self.head_size)
        self.use_layer_norm = use_layer_norm
        self.num_latents = num_latents
        self.latent_size = latent_size if latent_size is not None else hidden_size

        # Layer Normalization (applied first in forward pass if use_layer_norm is True)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, device=device, dtype=dtype)

        # Learnable latent vectors
        self.latents = nn.Parameter(torch.randn(num_latents, self.latent_size, device=device, dtype=dtype))

        # Latent projections
        self.latent_query = nn.Linear(self.latent_size, hidden_size, bias=bias, device=device, dtype=dtype)
        self.latent_value = nn.Linear(self.latent_size, hidden_size, bias=bias, device=device, dtype=dtype)

        # Input projections
        self.input_key = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
        self.input_value = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)

        # Output projections
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)

        # Dropout layers
        self.attention_dropout = nn.Dropout(dropout_p)
        self.output_dropout = nn.Dropout(dropout_p)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes the weights of the multi-latent attention layers."""
        # Initialize latent vectors with a normal distribution
        nn.init.normal_(self.latents, std=0.02)

        # Initialize linear projections
        nn.init.xavier_uniform_(self.latent_query.weight)
        nn.init.xavier_uniform_(self.latent_value.weight)
        nn.init.xavier_uniform_(self.input_key.weight)
        nn.init.xavier_uniform_(self.input_value.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        # Initialize biases to zero
        if self.latent_query.bias is not None:
            nn.init.zeros_(self.latent_query.bias)
        if self.latent_value.bias is not None:
            nn.init.zeros_(self.latent_value.bias)
        if self.input_key.bias is not None:
            nn.init.zeros_(self.input_key.bias)
        if self.input_value.bias is not None:
            nn.init.zeros_(self.input_value.bias)
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
            - [batch_size, seq_len, num_heads, head_size] and transposed if from_latents is False
            - [batch_size, num_latents, num_heads, head_size] and transposed if from_latents is True
        """
        if from_latents:
            # For latent vectors: [num_latents, hidden_size]
            x = x.view(self.num_latents, self.num_attention_heads, self.head_size)
            # Unsqueeze batch dimension: [1, num_latents, num_heads, head_size]
            return x.unsqueeze(0).transpose(1, 2)
        else:
            # For input sequences: [batch_size, seq_len, hidden_size]
            batch_size, seq_len, _ = x.size()
            x = x.view(batch_size, seq_len, self.num_attention_heads, self.head_size)
            # Transpose to [batch_size, num_heads, seq_len, head_size]
            return x.transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass for the multi-latent attention mechanism.

        Args:
            hidden_states: Input tensor, shape [batch_size, seq_len, hidden_size].
            attention_mask: Optional mask tensor, shape [batch_size, 1, 1, seq_len].
                           1 indicates positions to attend to, 0 indicates positions to mask out.

        Returns:
            Output tensor with shape [batch_size, seq_len, hidden_size].
        """
        # Save residual for later
        residual = hidden_states

        # Apply Layer Normalization if enabled (Pre-LN architecture)
        if self.use_layer_norm:
            hidden_states = self.layer_norm(hidden_states)

        batch_size, seq_len, _ = hidden_states.size()

        # Phase 1: Latents attend to input sequence
        # Project inputs to keys and values
        input_keys = self.input_key(hidden_states)  # [batch_size, seq_len, hidden_size]
        input_values = self.input_value(hidden_states)  # [batch_size, seq_len, hidden_size]

        # Project latents to queries
        latent_queries = self.latent_query(self.latents)  # [num_latents, hidden_size]

        # Reshape for multi-head attention
        input_keys = self._reshape_for_multihead_attention(input_keys)  # [batch_size, num_heads, seq_len, head_size]
        input_values = self._reshape_for_multihead_attention(
            input_values
        )  # [batch_size, num_heads, seq_len, head_size]
        latent_queries = self._reshape_for_multihead_attention(latent_queries, from_latents=True)
        # [batch_size, num_heads, num_latents, head_size]

        # Compute scaled dot-product attention: latent queries attend to input keys
        # [batch_size, num_heads, num_latents, head_size] x [batch_size, num_heads, head_size, seq_len]
        # -> [batch_size, num_heads, num_latents, seq_len]
        attention_scores = torch.matmul(latent_queries, input_keys.transpose(-1, -2)) / self.scaling

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to cover the latent dimension
            # [batch_size, 1, 1, seq_len] -> [batch_size, 1, num_latents, seq_len]
            expanded_mask = attention_mask.expand(-1, -1, self.num_latents, -1)
            attention_scores = attention_scores + expanded_mask

        # Normalize the attention scores to probabilities
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        # Apply dropout to the attention probabilities
        attention_probs = self.attention_dropout(attention_probs)

        # Compute the weighted sum of values
        # [batch_size, num_heads, num_latents, seq_len] x [batch_size, num_heads, seq_len, head_size]
        # -> [batch_size, num_heads, num_latents, head_size]
        latent_context = torch.matmul(attention_probs, input_values)

        # Phase 2: Transform updated latents
        # Reshape latent context: [batch_size, num_heads, num_latents, head_size] -> [batch_size, num_latents, hidden_size]
        latent_context = latent_context.transpose(1, 2).contiguous()
        latent_context = latent_context.view(batch_size, self.num_latents, self.hidden_size)

        # Project latents to output values
        latent_output_values = self.latent_value(latent_context)  # [batch_size, num_latents, hidden_size]

        # Reshape for final projection
        # Reshape to match the input sequence: [batch_size, seq_len, hidden_size]
        # We use a simple broadcasted expansion for simplicity
        # This effectively copies the same latent outputs to each position in the sequence
        output = latent_output_values.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)

        # Project to final representation
        output = self.out_proj(output)
        output = self.output_dropout(output)

        # Add residual connection
        output = residual + output

        return output


# Potential Extensions and Variations:
# - Implement cross-attention where queries come from one sequence and keys/values from another.
# - Add relative position embeddings to enhance attention with positional information.
# - Implement local attention variants that attend only to nearby tokens.
# - Create sparse attention mechanisms that selectively attend to a subset of tokens.
