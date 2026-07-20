import math

import torch
from torch import Tensor, nn

from llm.core.registry import register_attention, set_attention_kv_cache_capability
from llm.utils.common import make_factory_kwargs

from .sdpa import sdpa


@register_attention("mla")
class MultiLatentAttention(nn.Module):
    # MLA caches the K, V from ``input_kv_proj`` into the standard
    # ``KVCache`` / ``PagedKVCache`` pool — same cache contract as MHA.
    # The architectural caveat is that this is the *placeholder* MLA
    # (learnable latent queries, uniform-mean output broadcast over the
    # sequence). Real DeepSeek-V2-style MLA with latent-compressed K, V
    # and decoupled RoPE is a separate, larger slice.
    set_attention_kv_cache_capability("mla", supports=True)
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

    Note:
        This is the **placeholder** MLA — the latent queries attend to
        the full ``input_kv_proj(x)`` (no latent-dim compression, no
        decoupled RoPE). The output is a uniform average over the
        ``num_latents`` latent outputs broadcast to every sequence
        position, so the architectural benefit of per-position KV cache
        is limited; the cache only saves the ``input_kv_proj`` cost on
        incremental decode. DeepSeek-V2-style MLA (latent-compressed
        K, V, decoupled RoPE) is a separate slice.
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
        include_norm_residual: bool = True,
        num_kv_heads: int | None = None,
        window_size: int | None = None,
        num_latents: int = 16,
        latent_dim: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **_: object,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")

        factory_kwargs = make_factory_kwargs(device, dtype)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1 / math.sqrt(self.head_dim)
        self.norm_first = norm_first
        self.num_latents = num_latents
        self.latent_dim = latent_dim if latent_dim is not None else hidden_size
        self.is_causal = is_causal
        self.dropout_p = p
        self.include_norm_residual = include_norm_residual

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
        self.dropout = nn.Dropout(p)

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

    def _latent_attention(
        self,
        k: Tensor,
        v: Tensor,
        batch_size: int,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool,
    ) -> Tensor:
        """
        Process latent attention computation as a separate method for clarity.
        This function computes the attention between latent queries and input sequence.

        Args:
            k: Key tensor with shape [batch_size, num_heads, seq_len, head_dim]
            v: Value tensor with shape [batch_size, num_heads, seq_len, head_dim]
            batch_size: Batch size
            attn_mask: Optional attention mask.

                Incoming shape is the standard MHA convention
                ``[B, 1, S_q, S_k]`` where ``S_q`` is the new-token count
                and ``S_k`` is the cached context length. The latent
                attention's query axis is ``num_latents`` (latents are
                static parameters, not derived from the input), so we
                collapse ``S_q`` to the LAST position's mask (the
                canonical causal mask for the current generation step)
                and broadcast over ``num_latents``.

        Returns:
            Processed latent output with shape [batch_size, num_latents, hidden_size]
        """
        # Expand and project latent queries in one operation
        latent_q = self.latents.expand(batch_size, -1, -1)
        latent_q = self.latent_q_proj(latent_q.reshape(batch_size * self.num_latents, self.latent_dim))
        latent_q = latent_q.view(batch_size, self.num_latents, self.num_heads, self.head_dim)
        latent_q = latent_q.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_latents, head_dim]

        # Reshape the MHA-style mask ``[B, 1, S_q, S_k]`` into the latent
        # attention's mask ``[B, 1, num_latents, S_k]``. The latent queries
        # share the same key-visibility mask — take the last position's mask
        # (the "current token" view) and broadcast.
        if attn_mask is not None:
            if attn_mask.shape[2] != self.num_latents:
                attn_mask = attn_mask[:, :, -1:, :]
            attn_mask = attn_mask.expand(-1, -1, self.num_latents, -1)

        # Compute attention with conditional dropout during training
        latent_output = sdpa(
            query=latent_q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
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
        is_causal: bool | None = None,
        kv_cache: object | None = None,
        use_cache: bool = False,
        batch_indices: Tensor | None = None,
        start_pos: int | Tensor | None = None,
        paged_kv_cache: object | None = None,
        layer_idx: int | None = None,
    ) -> Tensor | tuple[Tensor, None]:
        """
        Optimized forward pass for the multi-latent attention mechanism.

        Args:
            hidden_states: Input tensor with shape [batch_size, seq_len, hidden_size].
            attn_mask: Optional mask tensor with shape [batch_size, 1, 1, seq_len].
                       1 indicates positions to attend to, 0 indicates positions to mask.
            kv_cache: Linear ``KVCache`` pool. Mutually exclusive with
                ``paged_kv_cache``.
            paged_kv_cache: Block-allocator ``PagedKVCache``. Mutually
                exclusive with ``kv_cache``.
            layer_idx: Required when ``paged_kv_cache`` is set; selects
                the per-layer K, V slice from the paged cache.

        Returns:
            Output tensor of shape ``[batch_size, seq_len, hidden_size]``.
            When ``use_cache=True``, returns ``(output, None)`` — the cache
            was updated in-place so there is nothing to return alongside
            the output (unlike MHA, which exposes the cached K, V).
        """
        if kv_cache is not None and paged_kv_cache is not None:
            raise ValueError("Pass either kv_cache or paged_kv_cache, not both.")

        use_causal = self.is_causal if is_causal is None else is_causal
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

        # KV cache routing — same parallel-parameter pattern as MHA. The
        # latent attention then runs over the (possibly cached) K, V.
        if paged_kv_cache is not None:
            # ``target_seq_len`` aligns the per-row paged gather with the
            # mask's key-axis (the engine builds its mask against the
            # model's ``max_seq_len``). Without this hint the gather pads
            # only to the per-batch max, which can be smaller than the
            # mask's k-axis when the engine's running sequences are
            # short.
            target_seq_len = attn_mask.shape[-1] if attn_mask is not None else None
            k, v = self._paged_kv_write(
                k=k,
                v=v,
                paged_kv_cache=paged_kv_cache,
                batch_indices=batch_indices,
                layer_idx=layer_idx,
                target_seq_len=target_seq_len,
            )
        elif kv_cache is not None:
            k, v = self._linear_kv_write(
                k=k,
                v=v,
                kv_cache=kv_cache,
                batch_indices=batch_indices,
                start_pos=start_pos,
            )

        # Process latent attention
        latent_output = self._latent_attention(k, v, batch_size, attn_mask, is_causal=use_causal)

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

        # Match the MHA contract:
        # - paged path returns the output tensor directly (the cache is
        #   mutated in place; the caller does not need a kv tuple);
        # - dense + use_cache returns ``(output, kv)``. For MLA the
        #   cached K, V are consumed internally by the latent attention,
        #   so we return ``None`` as the second element;
        # - no cache returns the output tensor.
        if paged_kv_cache is not None:
            return output
        if use_cache:
            return output, None
        return output

    def _linear_kv_write(
        self,
        k: Tensor,
        v: Tensor,
        kv_cache: object,
        batch_indices: Tensor | None,
        start_pos: int | Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Write the new K, V into the linear ``KVCache`` pool and return the cached view.

        For per-slot writes (``batch_indices`` is set) the cache contract
        is ``update_at_indices``; for a dense batch it's ``update``.
        """
        if batch_indices is not None:
            if start_pos is None:
                raise ValueError("start_pos must be provided when using batch_indices for KV cache update.")
            return kv_cache.update_at_indices(batch_indices, k, v, start_pos)
        return kv_cache.update(k, v)

    def _paged_kv_write(
        self,
        k: Tensor,
        v: Tensor,
        paged_kv_cache: object,
        batch_indices: Tensor | None,
        layer_idx: int | None,
        target_seq_len: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Write the new K, V into the paged cache and return the cached K, V slice.

        Per row: call ``paged_kv_cache.update(seq_id, k_b.T, v_b.T)``
        to append the new tokens. Then gather the per-row K, V via
        :meth:`PagedKVCache.get` so the latent attention runs over the
        full cached context (this is the cost the cache saves vs.
        recomputing ``input_kv_proj`` on every past token).

        Args:
            target_seq_len: Optional padding target for the returned K, V
                sequence axis. The latent attention's mask expects
                ``[B, 1, num_latents, target_seq_len]``; when set we pad
                the per-row K, V to this length with zeros (masked
                positions are out of range for the active sequence).
                When ``None``, we pad to the per-batch max seq length.

        Note:
            ``paged_attention_forward`` is *not* used here. That kernel
            returns the attended output — for MLA we need the raw K, V
            to feed the latent cross-attention block.
        """
        if layer_idx is None:
            raise ValueError(
                "layer_idx is required when paged_kv_cache is set; DecoderModel threads it through TransformerBlock."
            )
        if batch_indices is None:
            raise ValueError("batch_indices is required when paged_kv_cache is set.")

        # Per-row write into the paged cache. ``PagedKVCache.update``
        # expects ``[B, T, N_kv, D]`` (it transposes internally), so
        # transpose our ``[B, N_kv, T, D]`` k/v to match.
        seq_ids = batch_indices.tolist()
        for b, seq_id in enumerate(seq_ids):
            paged_kv_cache.update(
                seq_id=int(seq_id),
                k_new=k[b : b + 1].transpose(1, 2),
                v_new=v[b : b + 1].transpose(1, 2),
            )

        # Gather the per-row K, V via the public ``PagedKVCache.get`` API.
        # The latent attention expects ``[B, N_heads, T_total, head_dim]``
        # per row, padded with zeros for shorter sequences.
        batch_size, num_heads, _, head_dim = k.shape
        per_row_seq_lens = [paged_kv_cache.block_manager.get_num_tokens(int(sid)) for sid in seq_ids]
        if target_seq_len is None:
            target_seq_len = max(per_row_seq_lens) if per_row_seq_lens else 1
            target_seq_len = max(target_seq_len, 1)

        k_gathered = torch.zeros(
            batch_size,
            num_heads,
            target_seq_len,
            head_dim,
            device=k.device,
            dtype=k.dtype,
        )
        v_gathered = torch.zeros_like(k_gathered)
        for b, seq_id in enumerate(seq_ids):
            seq_len = per_row_seq_lens[b]
            if seq_len == 0:
                continue
            # ``PagedKVCache.get`` returns ``[N_kv, num_tokens, D]``;
            # the MLA contract is ``[B, N_heads, T_total, D]`` — the
            # per-row head count equals ``self.num_heads`` (no GQA in
            # the placeholder MLA).
            k_row, v_row = paged_kv_cache.get(int(seq_id), 0, seq_len)
            k_gathered[b, :, :seq_len] = k_row
            v_gathered[b, :, :seq_len] = v_row

        return k_gathered, v_gathered
