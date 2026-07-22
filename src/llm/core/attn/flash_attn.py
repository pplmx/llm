"""Flash Attention 2 attention implementation.

Registered as ``attn_impl="flash_attn"`` through ``ATTENTION_REGISTRY``.
The ``flash-attn`` package is an **optional dependency** — importing this
module never raises, but instantiating :class:`FlashAttention` does if
the package is not installed. This mirrors the soft-dependency contract
used elsewhere in the project (e.g. ``huggingface_hub`` in
``compat.hf_loader``).

The class itself follows the same surface as
:class:`llm.core.attn.MultiHeadAttention` so it can be substituted
transparently: same projection layers, same KV-cache integration, same
output projection. The only difference is the attention kernel:
``flash_attn.flash_attn_func`` instead of
``torch.nn.functional.scaled_dot_product_attention``.

On supported hardware (Ampere/Hopper) this is meaningfully faster for
training and (especially) long-context decode. On other devices the
fallback path through PyTorch SDPA in MHA is the right choice.
"""

from __future__ import annotations

import importlib.util

import torch
from torch import Tensor, nn

from llm.core.kv_cache import KVCache
from llm.core.registry import register_attention, set_attention_kv_cache_capability
from llm.utils.common import make_factory_kwargs

# Probe the optional dependency at import time so the registry entry is
# always populated (consumers can list names, validate configs, etc.)
# even when ``flash-attn`` is not installed. The actual gate fires at
# instantiation below.
_flash_attn_spec = importlib.util.find_spec("flash_attn")
FLASH_ATTN_AVAILABLE: bool = _flash_attn_spec is not None


@register_attention("flash_attn")
class FlashAttention(nn.Module):
    # KV-cache contract matches MHA: the wrapper writes K/V into the
    # shared ``KVCache`` pool during autoregressive decoding, so the
    # continuous batching engine and training engine work without
    # changes. ``ModelConfig.check_consistency`` consults this map.
    set_attention_kv_cache_capability("flash_attn", supports=True)
    """
    Flash Attention 2 wrapper exposing the standard MHA interface.

    Same projection layout as :class:`MultiHeadAttention` (combined
    QKV, output projection, optional pre-norm/residual). The attention
    kernel is ``flash_attn.flash_attn_func``; Q/K/V are reshaped from
    ``[B, N, S, D]`` to ``[B, S, H]`` for the call and the output is
    reshaped back.

    Args:
        hidden_size: Total dimension of the model.
        num_heads: Number of attention heads. Must divide ``hidden_size``.
        p: Dropout probability applied to attention weights and final
            output. (Forwarded to ``flash_attn_func``; ``flash-attn``
            applies dropout only during training.)
        bias: Whether to use bias in the QKV/output projections.
        eps: Epsilon for the optional pre-norm LayerNorm.
        norm_first: Whether to apply LayerNorm before (Pre-LN) or after
            (Post-LN) the attention block. Only used when
            ``include_norm_residual=True``.
        is_causal: Whether to apply causal masking by default.
        include_norm_residual: Whether this block owns its pre-norm and
            residual connection (same convention as ``mha``).
        num_kv_heads: For GQA/MQA. Defaults to ``num_heads``. Must
            divide ``num_heads``.
        device: Target device for parameters.
        dtype: Target dtype for parameters.

    Raises:
        ImportError: When constructed without ``flash-attn`` installed.
            Install with ``pip install flash-attn`` (builds a CUDA
            wheel; not available on plain PyPI for some platforms —
            see the project's ``[perf]`` extra).
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
        include_norm_residual: bool = True,
        num_kv_heads: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if not FLASH_ATTN_AVAILABLE:
            raise ImportError(
                "FlashAttention requires the optional 'flash-attn' package. "
                "Install it with `pip install 'llm[perf]'` "
                "(or `pip install flash-attn` directly on a CUDA host)."
            )

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")

        factory_kwargs = make_factory_kwargs(device, dtype)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.norm_first = norm_first
        self.is_causal = is_causal
        self.p = p
        self.include_norm_residual = include_norm_residual

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})")

        self.norm = None
        if self.include_norm_residual:
            self.norm = nn.LayerNorm(hidden_size, eps=eps, **factory_kwargs)

        self.qkv_dim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv_proj = nn.Linear(hidden_size, self.qkv_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(p)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform init for projections, zero bias."""
        for proj in (self.qkv_proj, self.out_proj):
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
        """Forward pass.

        See :meth:`MultiHeadAttention.forward` for the full argument
        contract — the signatures are intentionally identical so the
        continuous batching engine and training engine can pick
        ``flash_attn`` as a drop-in replacement.

        Note:
            ``attn_mask`` is **ignored** — ``flash_attn_func`` does not
            accept arbitrary masks. For non-causal masking or padded
            sequences use ``is_causal=False`` and pre-pad, or fall back
            to ``attn_impl="mha"``. Sliding-window / padded-mask
            support requires ``flash_attn_varlen_func`` (future work).
            ``paged_kv_cache`` is also rejected on this path — flash-attn
            does not expose a paged-attn kernel; use ``attn_impl="mha"``
            when serving with paged KV.

            ``prefix_kv`` is supported (Li & Liang 2021). The prefix
            K/V are concatenated to the projected K/V after the
            KV-cache write (so the cache stores only dynamic tokens)
            and before the GQA repeat (so the prefix is treated like a
            regular token and replicated to all query heads). Prefix
            dtype is auto-cast to the projected K/V dtype to satisfy
            ``flash_attn_func``'s fp16/bf16 requirement. The captured
            ``current_kv`` (when ``use_cache=True``) excludes the
            prefix — matching the MHA contract at
            ``MultiHeadAttention.forward``.
        """
        if paged_kv_cache is not None:
            raise NotImplementedError(
                "FlashAttention does not support paged_kv_cache; "
                "use attn_impl='mha' (which routes through paged_attention_forward)."
            )
        # Local import so an uninstalled ``flash-attn`` does not crash
        # the package at import time (we already gated on
        # ``FLASH_ATTN_AVAILABLE`` in ``__init__``).
        from flash_attn import flash_attn_func

        batch_size, seq_len, _ = hidden_states.size()
        use_causal = self.is_causal if is_causal is None else is_causal

        if self.include_norm_residual and self.norm is not None:
            residual = hidden_states
            x_for_qkv = self.norm(hidden_states) if self.norm_first else hidden_states
        else:
            x_for_qkv = hidden_states

        # Project + split QKV (identical to MHA).
        qkv = self.qkv_proj(x_for_qkv)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

        # Reshape: [B, S, N*D] -> [B, N, S, D]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        has_past = False
        if kv_cache is not None:
            if batch_indices is not None:
                if start_pos is None:
                    raise ValueError("start_pos must be provided when using batch_indices for KV cache update.")
                k, v = kv_cache.update_at_indices(batch_indices, k, v, start_pos)
                has_past = True
            else:
                k, v = kv_cache.update(k, v)
                has_past = kv_cache.seq_len > seq_len

        if use_cache:
            current_kv = (k, v)

        # Prefix injection (Li & Liang 2021): prepend ``prefix_kv`` to
        # the projected K/V so the new tokens attend to the prefix in
        # addition to the cached context. Done AFTER the KV-cache write
        # (so the cache only stores dynamic tokens) and BEFORE the GQA
        # repeat (so the prefix is treated like a regular token and
        # replicated to all query heads).
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
            if prefix_k.shape[0] != batch_size:
                raise ValueError(f"prefix batch ({prefix_k.shape[0]}) must match hidden_states batch ({batch_size})")
            # flash_attn_func requires fp16/bf16. Cast the prefix to
            # the projected K/V dtype to satisfy the kernel contract.
            target_dtype = k.dtype
            if prefix_k.dtype != target_dtype:
                prefix_k = prefix_k.to(target_dtype)
            if prefix_v.dtype != target_dtype:
                prefix_v = prefix_v.to(target_dtype)
            k = torch.cat([prefix_k, k], dim=2)
            v = torch.cat([prefix_v, v], dim=2)

        # GQA: replicate K/V across query heads.
        if self.num_kv_heads != self.num_heads:
            num_queries_per_kv = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(num_queries_per_kv, dim=1)
            v = v.repeat_interleave(num_queries_per_kv, dim=1)

        # flash_attn_func expects [B, S, H] (heads merged into the last
        # axis). Transpose and make contiguous (the underlying CUDA
        # kernel does not accept strided inputs in this layout).
        q_bsh = q.transpose(1, 2).contiguous()
        k_bsh = k.transpose(1, 2).contiguous()
        v_bsh = v.transpose(1, 2).contiguous()

        # flash_attn_func requires fp16/bf16. We keep the call guarded
        # by the caller (MHA does the same — float dtype falls back via
        # the engine layer). Apply causal only when there is no past
        # context: with KV cache populated, the cache handles positions.
        attn_output = flash_attn_func(
            q_bsh,
            k_bsh,
            v_bsh,
            dropout_p=self.p if self.training else 0.0,
            causal=use_causal if not has_past else False,
        )

        # Output back to [B, N, S, D].
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # [B, N, S, D] -> [B, S, H]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        projected_output = self.dropout(self.out_proj(attn_output))

        if self.include_norm_residual and self.norm is not None:
            output = residual + projected_output
            if not self.norm_first:
                output = self.norm(output)
        else:
            output = projected_output

        if use_cache:
            return output, current_kv
        return output
