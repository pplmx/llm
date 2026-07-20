from __future__ import annotations

import torch
from torch import Tensor, nn

from llm.core.kv_cache import KVCache
from llm.core.paged_attention.attention import paged_attention_forward
from llm.core.registry import register_attention, set_attention_kv_cache_capability
from llm.utils.common import make_factory_kwargs

from .sdpa import sdpa


@register_attention("mha")
class MultiHeadAttention(nn.Module):
    # Standard MHA writes into the KV-cache pool during autoregressive decoding.
    # The continuous batching engine and training engine both depend on this.
    set_attention_kv_cache_capability("mha", supports=True)
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

        factory_kwargs = make_factory_kwargs(device, dtype)
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
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
        batch_indices: Tensor | None = None,
        start_pos: int | Tensor | None = None,
        paged_kv_cache: object | None = None,
        layer_idx: int | None = None,
        prefix_kv: tuple[Tensor, Tensor] | None = None,
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
            kv_cache (KVCache | None): Pre-allocated KV cache for efficient autoregressive generation.
                When provided, updates are done in-place without memory allocation.
            prefix_kv (tuple[Tensor, Tensor] | None): Optional prefix K/V to prepend to the
                projected K/V before the attention compute. Used by the Prefix Tuning slice
                (T2 PEFT) — see :class:`llm.core.prefix_tuning.PrefixTuningAttention`. Tensors
                must be shape ``[B, num_kv_heads, prefix_len, head_dim]``. Injected **after**
                the KV cache write so the cache only stores dynamic tokens.
            use_cache (bool): Whether to return the updated (key, value) pair.
            batch_indices (Tensor | None): Indices for specific KV cache slots [B]. Use with update_at_indices.
            start_pos (int | Tensor | None): Explicit write position for cache update. required if batch_indices is used.
            paged_kv_cache (PagedKVCache | None): Block-allocator KV cache. When set
                the linear ``kv_cache`` argument is ignored — the model writes K/V
                into the paged blocks and reads via ``paged_attention_forward``.
                ``batch_indices`` doubles as the per-row ``seq_id`` (the engine
                passes slot ids that we treat as ``PagedKVCache`` sequence ids).
            layer_idx (int | None): Index of this block in the decoder. Required
                when ``paged_kv_cache`` is set; used to slice the per-layer K/V
                tensor out of ``PagedKVCache.k_cache[layer_idx]``.

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

        # KV Cache handling
        if paged_kv_cache is not None:
            return self._forward_paged(
                q=q,
                k=k,
                v=v,
                paged_kv_cache=paged_kv_cache,
                batch_indices=batch_indices,
                layer_idx=layer_idx,
                residual=residual if self.include_norm_residual and self.norm is not None else None,
            )

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

        if use_cache:
            current_kv = (k, v)

        # GQA: Repeat K, V if needed
        # Prefix injection happens BEFORE the GQA repeat so the prefix is
        # treated exactly like a regular token (repeated num_queries_per_kv
        # times across the query heads).
        if prefix_kv is not None:
            prefix_k, prefix_v = prefix_kv
            if prefix_k.shape != prefix_v.shape:
                raise ValueError(
                    f"prefix_k and prefix_v must share shape; got {tuple(prefix_k.shape)} vs {tuple(prefix_v.shape)}"
                )
            if prefix_k.shape[1] != self.num_kv_heads:
                raise ValueError(
                    f"prefix num_kv_heads ({prefix_k.shape[1]}) must match attention num_kv_heads ({self.num_kv_heads})"
                )
            if prefix_k.shape[3] != self.head_dim:
                raise ValueError(
                    f"prefix head_dim ({prefix_k.shape[3]}) must match attention head_dim ({self.head_dim})"
                )
            k = torch.cat([prefix_k, k], dim=2)
            v = torch.cat([prefix_v, v], dim=2)

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

    def _forward_paged(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        paged_kv_cache: object,
        batch_indices: Tensor | None,
        layer_idx: int | None,
        residual: Tensor | None,
    ) -> Tensor:
        """Run the attention computation through a :class:`PagedKVCache`.

        Per-row write: ``paged_kv_cache.update(seq_id, k_b.T, v_b.T)``
        appends the new tokens to that sequence's block table. The
        sequence id is taken from ``batch_indices`` (the engine passes
        slot ids; ``PagedKVCache`` treats them as sequence ids). After
        the writes we read the per-row block tables / seq lengths and
        call :func:`paged_attention_forward` to compute attention over
        the gathered context.

        Args:
            q: Projected query tensor ``[B, N_q, S, D]``.
            k: Projected key tensor ``[B, N_kv, S, D]``.
            v: Projected value tensor ``[B, N_kv, S, D]``.
            paged_kv_cache: The block-allocator cache (typed ``object``
                to avoid a circular import on ``core.paged_attention``).
            batch_indices: Slot ids per row ``[B]``; doubles as the
                ``seq_id`` for ``PagedKVCache.update``.
            layer_idx: Index of this block in the decoder; slices
                ``paged_kv_cache.k_cache[layer_idx]``.
            residual: Pre-norm residual tensor (``None`` when this
                block does not own the residual).

        Returns:
            Attention output ``[B, S, H]`` after output projection.
        """
        if layer_idx is None:
            raise ValueError(
                "layer_idx is required when paged_kv_cache is set; DecoderModel threads it through TransformerBlock."
            )
        if batch_indices is None:
            raise ValueError(
                "batch_indices is required when paged_kv_cache is set; the engine passes slot ids per row."
            )

        batch_size, _, seq_len, _ = q.shape

        # 1. Per-row write into the paged cache. ``PagedKVCache.update``
        #    expects ``[B, T, N_kv, D]`` (it transposes internally), so
        #    transpose our ``[B, N_kv, T, D]`` k/v to match.
        seq_ids = batch_indices.tolist()
        for b, seq_id in enumerate(seq_ids):
            paged_kv_cache.update(
                seq_id=int(seq_id),
                k_new=k[b : b + 1].transpose(1, 2),
                v_new=v[b : b + 1].transpose(1, 2),
            )

        # 2. Build ``block_tables`` and ``seq_lens`` per row from the
        #    BlockManager's view of each sequence.
        block_size = paged_kv_cache.block_size
        max_blocks = max(
            (len(paged_kv_cache.get_block_table(int(sid))) for sid in seq_ids),
            default=1,
        )
        # Pad block-table columns to a single tensor shape.
        block_tables = torch.full((batch_size, max_blocks), -1, dtype=torch.long, device=q.device)
        seq_lens = torch.zeros(batch_size, dtype=torch.long, device=q.device)
        for b, seq_id in enumerate(seq_ids):
            table = paged_kv_cache.get_block_table(int(seq_id))
            block_tables[b, : len(table)] = torch.tensor(table, dtype=torch.long)
            seq_lens[b] = paged_kv_cache.block_manager.get_num_tokens(int(seq_id))

        # 3. Run the paged attention kernel over the per-layer slice.
        k_layer = paged_kv_cache.k_cache[layer_idx]
        v_layer = paged_kv_cache.v_cache[layer_idx]
        attn_output = paged_attention_forward(
            q=q,
            k_cache=k_layer,
            v_cache=v_layer,
            block_tables=block_tables,
            seq_lens=seq_lens,
            num_kv_heads=self.num_kv_heads,
            block_size=block_size,
        )  # [B, N_q, S, D]

        # 4. Reshape and project — same post-processing as the linear path.
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        projected_output = self.dropout(self.out_proj(attn_output))

        if self.include_norm_residual and self.norm is not None and residual is not None:
            output = residual + projected_output
            if not self.norm_first:
                output = self.norm(output)
        else:
            output = projected_output
        return output
