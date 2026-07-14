from __future__ import annotations

import torch
import torch.nn as nn

from llm.core.kv_cache import KVCache
from llm.utils.common import make_factory_kwargs


class TransformerBlock(nn.Module):
    """
    A single Transformer block, comprising a Multi-Head Attention (MHA) layer
    and a Multi-Layer Perceptron (MLP) layer, with normalization and residual connections.

    The block can be configured for Pre-LN (Layer Normalization before sublayer)
    or Post-LN (Layer Normalization after sublayer and residual connection).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int | None = None,
        attn_dropout_p: float = 0.1,
        mlp_dropout_p: float = 0.1,
        mlp_activation: str | nn.Module = "gelu",
        norm_eps: float = 1e-5,
        norm_first: bool = True,
        is_causal: bool = False,  # Default causality for MHA within this block
        qkv_bias: bool = True,  # Bias for QKV projections in MHA
        mlp_bias: bool = True,  # Bias for Linear layers in MLP
        num_experts: int = 0,
        top_k: int = 0,
        num_kv_heads: int | None = None,  # For GQA support
        use_glu: bool = False,  # New: For SwiGLU support
        norm_type: type[nn.Module] | nn.Module = nn.LayerNorm,
        window_size: int | None = None,  # Sliding window attention
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        # Registry keys
        attn_impl: str = "mha",
        mlp_impl: str = "mlp",
    ):
        """
        Initializes the TransformerBlock.
        """
        super().__init__()
        factory_kwargs = make_factory_kwargs(device, dtype)

        from llm.core.registry import ATTENTION_REGISTRY, MLP_REGISTRY

        self.norm_first = norm_first
        self.hidden_size = hidden_size

        # Initialize Norms. ``norm_type`` must be a factory callable
        # ``(**kwargs) -> nn.Module`` — typically one of the entries in
        # ``NORM_REGISTRY``. Already-instantiated ``nn.Module`` instances are
        # rejected because the previous isinstance(type) branch that
        # deep-copied them was a code smell (Finding C).
        if isinstance(norm_type, nn.Module):
            raise TypeError(
                "norm_type must be a factory callable (e.g. from NORM_REGISTRY), "
                "not an already-constructed nn.Module. Pass norm_impl='rms_norm' "
                "or 'layer_norm' to DecoderModel instead of pre-constructed norm modules."
            )
        if not callable(norm_type):
            raise TypeError(
                f"norm_type must be a callable factory, got {type(norm_type).__name__}."
            )
        self.norm1 = norm_type(hidden_size, eps=norm_eps, **factory_kwargs)
        self.norm2 = norm_type(hidden_size, eps=norm_eps, **factory_kwargs)

        # Initialize Attention via Registry
        attn_cls = ATTENTION_REGISTRY.get(attn_impl)
        self.self_attn = attn_cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            p=attn_dropout_p,
            bias=qkv_bias,
            is_causal=is_causal,
            include_norm_residual=False,
            eps=norm_eps,
            norm_first=False,
            num_kv_heads=num_kv_heads,
            window_size=window_size,
            **factory_kwargs,
        )

        # Initialize MLP via Registry
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        mlp_cls = MLP_REGISTRY.get(mlp_impl)

        # Prepare kwargs for MLP/MoE
        # Note: Different implementations might need different kwargs.
        # Ideally we pass a config object, but here we pass common args.
        # MoE needs num_experts and top_k, MLP doesn't.
        # We pass them as **kwargs, assuming constructors handle extra args or we filter.
        # But our classes strictly define __init__.
        # So we construct specific kwargs map.

        common_mlp_kwargs = {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "activation": mlp_activation,
            "dropout_p": mlp_dropout_p,
            "bias": mlp_bias,
            "norm_eps": norm_eps,
            **factory_kwargs,
        }

        if mlp_impl == "moe":
            # Add MoE specific args
            if num_experts <= 0 or top_k <= 0:
                raise ValueError("num_experts and top_k must be positive for MoE.")
            common_mlp_kwargs["num_experts"] = num_experts
            common_mlp_kwargs["top_k"] = top_k
        else:
            # Add MLP specific args (standard MLP doesn't need num_experts)
            common_mlp_kwargs["include_norm_residual"] = False
            common_mlp_kwargs["use_glu"] = use_glu
            common_mlp_kwargs["norm_first"] = False

        self.mlp = mlp_cls(**common_mlp_kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool | None = None,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
        batch_indices: torch.Tensor | None = None,
        start_pos: int | torch.Tensor | None = None,
        paged_kv_cache: object | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the Transformer block.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [B, S, H].
            attn_mask (torch.Tensor, optional): Attention mask for MHA.
            is_causal (bool, optional): Overrides the default MHA causality for this pass.
                                        If None, MHA's default `is_causal` is used.
            kv_cache (KVCache | None): Pre-allocated KV cache for efficient autoregressive generation.
            use_cache (bool): Whether to return the updated (key, value) pair.
            batch_indices(torch.Tensor | None): Cache update indices.
            start_pos (int | torch.Tensor | None): Cache update position.
            paged_kv_cache (object | None): Block-allocator KV cache; ignored
                when ``None``. When set, ``kv_cache`` is unused and ``layer_idx``
                must point at this block's index in the decoder.
            layer_idx (int | None): Index of this block in the decoder; required
                when ``paged_kv_cache`` is set.

        Returns:
            torch.Tensor or tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
                - If use_cache=False: Output tensor of shape [B, S, H].
                - If use_cache=True: (Output tensor, (current_key, current_value))
        """
        # Determine causality for the MHA call
        # If is_causal is provided as an argument, it overrides the MHA's default.
        # Otherwise, MHA uses its own self.is_causal.
        # The MHA forward method handles this logic if is_causal=None is passed.

        residual = hidden_states

        # 1. Multi-Head Attention Sublayer
        if self.norm_first:
            hidden_states = self.norm1(hidden_states)

        attn_outputs = self.self_attn(
            hidden_states,
            attn_mask=attn_mask,
            is_causal=is_causal,
            kv_cache=kv_cache,
            use_cache=use_cache,
            batch_indices=batch_indices,
            start_pos=start_pos,
            paged_kv_cache=paged_kv_cache,
            layer_idx=layer_idx,
        )

        if paged_kv_cache is not None:
            # Paged path returns the output directly (no separate kv
            # tuple to surface — the cache is mutated in place).
            attn_output = attn_outputs
            current_kv = None
        elif use_cache:
            attn_output, current_kv = attn_outputs
        else:
            attn_output = attn_outputs

        # Apply residual connection
        # Pre-LN MHA: output = residual + Attention(Norm(x))
        # Post-LN MHA: output = Norm(residual + Attention(x))
        if self.norm_first:
            hidden_states = residual + attn_output
            residual = hidden_states  # Update residual for next block
        else:
            hidden_states = self.norm1(residual + attn_output)
            residual = hidden_states  # Update residual for next block

        # 2. MLP Sublayer
        if self.norm_first:
            hidden_states = self.norm2(hidden_states)

        mlp_output = self.mlp(hidden_states)

        # Apply residual connection
        output = residual + mlp_output if self.norm_first else self.norm2(residual + mlp_output)

        if paged_kv_cache is not None:
            # The paged cache is mutated in place; there is no per-block
            # KV tuple to surface to the caller.
            return output
        if use_cache:
            return output, current_kv
        return output
