from __future__ import annotations

import torch
from torch import Tensor, nn

from llm.core.kv_cache import KVCache
from llm.core.registry import ATTENTION_REGISTRY

from .sdpa import sdpa


@ATTENTION_REGISTRY.register("mha")
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA) mechanism.

    Integrates Layer Normalization and residual connection, supporting Pre-LN and Post-LN modes.

    Args:
        hidden_size (int): Total dimension of the model.
        num_heads (int): Number of attention heads. Must divide hidden_size. Defaults to 8.
        p (float): Dropout probability applied to attention weights and final output. Defaults to 0.1.
        bias (bool): Whether to use bias in the linear layers (QKV projection and output projection). Defaults to False.
        eps (float): Epsilon value for Layer Normalization. Defaults to 1e-5.
        norm_first (bool): Whether to use Pre-LN (True) or Post-LN (False) architecture. Defaults to True.
        is_causal (bool): Whether to apply causal masking by default (e.g., for decoders). Defaults to False.
        device (torch.device | str | None): Target device for model parameters. Defaults to None (inferred).
        dtype (torch.dtype | None): Target data type for model parameters. Defaults to None (inferred).

    Attributes:
        head_dim (int): Dimension of each attention head.
        qkv_proj (nn.Linear): Combined Q, K, V projection layer.
        out_proj (nn.Linear): Output projection layer.
        norm (nn.LayerNorm): Layer normalization module.
        dropout (nn.Dropout): Dropout layer applied after the output projection.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        p: float = 0.1,
        bias: bool = False,
        eps: float = 1e-5,
        norm_first: bool = True,
        is_causal: bool = False,
        include_norm_residual: bool = True,  # New parameter
        num_kv_heads: int | None = None,  # New: For GQA/MQA support
        window_size: int | None = None,  # Sliding window attention
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.norm_first = norm_first  # Relevant only if include_norm_residual is True
        self.is_causal = is_causal
        self.p = p
        self.include_norm_residual = include_norm_residual
        self.window_size = window_size

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})")

        self.norm = None
        if self.include_norm_residual:
            self.norm = nn.LayerNorm(hidden_size, eps=eps, **factory_kwargs)

        self.qkv_dim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv_proj = nn.Linear(hidden_size, self.qkv_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(p)  # This is for the output projection

        self._init_weights()

    def _init_weights(self):
        """Initialize linear layer weights (Xavier uniform) and biases (zeros)."""
        for proj in [self.qkv_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        kv_cache: KVCache | None = None,  # noqa: F821
        use_cache: bool = False,
        batch_indices: Tensor | None = None,
        start_pos: int | Tensor | None = None,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Forward pass.

        Args:
            hidden_states (Tensor): Input tensor of shape [B, S, H] (Batch, Sequence Length, Hidden Size).
            attn_mask (Tensor | None): Optional attention mask.
                - For F.scaled_dot_product_attention, expected to be a boolean tensor where `True` indicates masking.
                - Shape should be broadcastable to [B, N, S, S] (Batch, Num Heads, Seq Len, Seq Len).
                - E.g., Padding mask could be [B, 1, 1, S] or [B, 1, S, S].
            is_causal (bool | None): Whether to enforce causal masking for this forward pass.
                - If `None` (default), uses the default `self.is_causal` set during initialization.
                - If `True` or `False`, overrides the default setting.
            past_key_value (tuple[Tensor, Tensor] | None): [DEPRECATED] Tuple of (key, value) from previous steps.
                Each has shape [B, N, S_prev, D]. Use `kv_cache` for better performance.
            kv_cache (KVCache | None): Pre-allocated KV cache for efficient autoregressive generation.
                When provided, updates are done in-place without memory allocation.
            use_cache (bool): Whether to return the updated (key, value) pair.
            batch_indices (Tensor | None): Indices for specific KV cache slots [B]. Use with update_at_indices.
            start_pos (int | Tensor | None): Explicit write position for cache update. required if batch_indices is used.

        Returns:
            Tensor or tuple[Tensor, tuple[Tensor, Tensor]]:
                - If use_cache=False: Output tensor of shape [B, S, H].
                - If use_cache=True: (Output tensor, (current_key, current_value))
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Determine causality for this call
        use_causal = self.is_causal if is_causal is None else is_causal

        # Prepare input for QKV projection
        # If norm and residual are handled by this module, apply norm first (if pre-norm)
        if self.include_norm_residual and self.norm is not None:
            residual = hidden_states
            x_for_qkv = self.norm(hidden_states) if self.norm_first else hidden_states
        else:
            # If no norm/residual by this module, use hidden_states directly
            # No residual variable needed here if not added by this module
            x_for_qkv = hidden_states

        # 2. Project Q, K, V and reshape
        qkv = self.qkv_proj(x_for_qkv)  # [B, S, (N_q + 2*N_kv) * D]

        # Split Q, K, V
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

        # Reshape and transpose for attention calculation: [B, N, S, D]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # KV Cache handling: prefer KVCache object, fallback to legacy tuple
        has_past = False
        if kv_cache is not None:
            # Use efficient pre-allocated cache (in-place update)
            if batch_indices is not None:
                if start_pos is None:
                    raise ValueError("start_pos must be provided when using batch_indices for KV cache update.")
                k, v = kv_cache.update_at_indices(batch_indices, k, v, start_pos)
                # has_past logic for 'update_at_indices' scenario:
                # It implies we are manually managing positions, so usually we don't rely on global seq_len check?
                # SDPA 'is_causal' logic:
                # If we are in Decode (seq_len=1), is_causal=False usually (we attend to all past).
                # If we are in Prefill, is_causal=True.
                # Let's assume the caller sets is_causal correctly or we are managing 'has_past' effectively.
                # For 'update_at_indices', we likely have past data.
                has_past = True
            else:
                k, v = kv_cache.update(k, v)
                has_past = kv_cache.seq_len > seq_len
        elif past_key_value is not None:
            # Legacy: use torch.cat (allocates memory each step)
            prev_k, prev_v = past_key_value
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
            has_past = True

        if use_cache:
            current_kv = (k, v)

        # GQA: Repeat K, V if needed
        if self.num_kv_heads != self.num_heads:
            # k, v: [B, N_kv, S, D] -> [B, N_q, S, D]
            num_queries_per_kv = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(num_queries_per_kv, dim=1)
            v = v.repeat_interleave(num_queries_per_kv, dim=1)

        # 3. Attention computation
        # Use common SDPA wrapper to handle mask polarity and window size
        attn_output = sdpa(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.p if self.training else 0.0,
            is_causal=use_causal if not has_past else False,
            scale=None,
            window_size=self.window_size,
        )  # Output shape: [B, N, S, D]

        # 4. Combine head outputs
        # [B, N, S, D] -> [B, S, N, D] -> [B, S, H]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)

        # 5. Output projection and dropout
        projected_output = self.dropout(self.out_proj(attn_output))

        if self.include_norm_residual and self.norm is not None:
            # 6. Residual connection
            output = residual + projected_output

            # 7. Layer Normalization (Post-LN mode)
            if not self.norm_first:
                output = self.norm(output)
        else:
            # No residual, no norm by this module
            output = projected_output

        if use_cache:
            return output, current_kv
        return output
