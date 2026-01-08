from __future__ import annotations

import torch
import torch.nn as nn

from llm.core.kv_cache import KVCache


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
        use_moe: bool = False,  # New: Whether to use MoE instead of MLP
        num_experts: int = 0,  # New: Number of experts if use_moe is True
        top_k: int = 0,  # New: Number of top experts to select if use_moe is True
        num_kv_heads: int | None = None,  # New: For GQA support
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
        factory_kwargs = {"device": device, "dtype": dtype}

        from llm.core.registry import ATTENTION_REGISTRY, MLP_REGISTRY

        self.norm_first = norm_first
        self.hidden_size = hidden_size

        # Initialize Norms
        if isinstance(norm_type, type):
            self.norm1 = norm_type(hidden_size, eps=norm_eps, **factory_kwargs)
            self.norm2 = norm_type(hidden_size, eps=norm_eps, **factory_kwargs)
        else:
            import copy

            self.norm1 = copy.deepcopy(norm_type)
            self.norm2 = copy.deepcopy(norm_type)
            if device is not None or dtype is not None:
                self.norm1 = self.norm1.to(device=device, dtype=dtype)
                self.norm2 = self.norm2.to(device=device, dtype=dtype)

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
        # Support legacy use_moe arg if passed, but prefer mlp_impl
        if use_moe:
            mlp_impl = "moe"

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
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the Transformer block.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [B, S, H].
            attn_mask (torch.Tensor, optional): Attention mask for MHA.
            is_causal (bool, optional): Overrides the default MHA causality for this pass.
                                        If None, MHA's default `is_causal` is used.
            past_key_value (tuple[Tensor, Tensor] | None): [DEPRECATED] Tuple of (key, value) from previous steps.
            kv_cache (KVCache | None): Pre-allocated KV cache for efficient autoregressive generation.
            use_cache (bool): Whether to return the updated (key, value) pair.

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
            is_causal=is_causal,  # Pass through, MHA handles None
            past_key_value=past_key_value,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )

        if use_cache:
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

        if use_cache:
            return output, current_kv
        return output


if __name__ == "__main__":
    # Example Usage (for self-testing, not part of the module's core logic)
    device_ex = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_ex = torch.float32

    hidden_size_ex = 64
    num_heads_ex = 8
    batch_size_ex = 4
    seq_len_ex = 10

    print(f"TransformerBlock Example running on device: {device_ex}, dtype: {dtype_ex}")

    # Dummy input
    dummy_input = torch.randn(batch_size_ex, seq_len_ex, hidden_size_ex, device=device_ex, dtype=dtype_ex)

    # Pre-LN configuration
    print("\nTesting Pre-LN TransformerBlock...")
    pre_ln_block = TransformerBlock(
        hidden_size=hidden_size_ex, num_heads=num_heads_ex, norm_first=True, device=device_ex, dtype=dtype_ex
    )
    pre_ln_block.eval()  # For consistent dropout behavior if any were active by default
    output_pre_ln = pre_ln_block(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape (Pre-LN): {output_pre_ln.shape}")
    assert output_pre_ln.shape == dummy_input.shape

    # Post-LN configuration
    print("\nTesting Post-LN TransformerBlock...")
    post_ln_block = TransformerBlock(
        hidden_size=hidden_size_ex, num_heads=num_heads_ex, norm_first=False, device=device_ex, dtype=dtype_ex
    )
    post_ln_block.eval()
    output_post_ln = post_ln_block(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape (Post-LN): {output_post_ln.shape}")
    assert output_post_ln.shape == dummy_input.shape

    # Test with causal masking
    print("\nTesting Pre-LN TransformerBlock with is_causal=True...")
    pre_ln_causal_block = TransformerBlock(
        hidden_size=hidden_size_ex,
        num_heads=num_heads_ex,
        norm_first=True,
        is_causal=True,  # Block default causality
        device=device_ex,
        dtype=dtype_ex,
    )
    pre_ln_causal_block.eval()
    output_pre_ln_causal = pre_ln_causal_block(dummy_input)  # Uses block's default
    print(f"  Output shape (Pre-LN, Causal default): {output_pre_ln_causal.shape}")
    assert output_pre_ln_causal.shape == dummy_input.shape

    # Override causality in forward pass
    output_pre_ln_non_causal_override = pre_ln_causal_block(dummy_input, is_causal=False)
    print(f"  Output shape (Pre-LN, Causal override to False): {output_pre_ln_non_causal_override.shape}")
    assert output_pre_ln_non_causal_override.shape == dummy_input.shape

    # Test with attention mask
    # Example: padding mask - True means masked position in F.scaled_dot_product_attention
    # For MHA, a common padding mask shape is [B, 1, 1, S_key] or broadcastable
    # For this test, let's make a simple mask that SDPA can use.
    # Mask out the last token for each sequence in the batch.
    attn_mask_ex = torch.zeros(batch_size_ex, seq_len_ex, seq_len_ex, device=device_ex, dtype=torch.bool)
    attn_mask_ex[:, :, -1] = True  # Mask the last key for all queries
    # SDPA expects mask shape like [B, N, S_q, S_k] or [B, 1, S_q, S_k] or [B, 1, 1, S_k] for padding mask
    # For simplicity, we'll pass a mask that is [B, S_q, S_k] and SDPA should broadcast it over heads.
    # Or, more correctly for MHA, it should be [B, N, S, S] or [B, 1, S, S].
    # The MHA layer's forward expects a mask that works with F.scaled_dot_product_attention.
    # A [B, S, S] mask will be broadcast to [B, 1, S, S] by SDPA.
    # My current MHA takes [B, N, S, S] or [B, 1, 1, S] or [B, 1, S, S]
    # For this example, let's assume the mask is prepared correctly for MHA.
    # For testing purposes, a [B, S, S] mask that SDPA can broadcast is fine.
    # Let's use a more MHA-friendly mask structure: [B, 1, S, S]
    attn_mask_mha_ex = torch.zeros(batch_size_ex, 1, seq_len_ex, seq_len_ex, device=device_ex, dtype=torch.bool)
    attn_mask_mha_ex[:, :, :, -1] = True  # Mask last key for all queries for all heads

    print("\nTesting Pre-LN TransformerBlock with attention_mask...")
    output_pre_ln_masked = pre_ln_block(dummy_input, attn_mask=attn_mask_mha_ex)
    print(f"  Output shape (Pre-LN, Masked): {output_pre_ln_masked.shape}")
    assert output_pre_ln_masked.shape == dummy_input.shape

    print("\nAll basic __main__ tests passed.")
