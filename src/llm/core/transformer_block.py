import torch
import torch.nn as nn

from llm.core.attn.mha import MultiHeadAttention
from llm.core.mlp import MLP
from llm.core.moe.moe import MoE # Import MoE


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
        mlp_intermediate_size: int | None = None,
        attn_dropout_p: float = 0.1,
        mlp_dropout_p: float = 0.1,
        mlp_activation: str | nn.Module = "gelu",
        norm_eps: float = 1e-5,
        norm_first: bool = True,
        is_causal: bool = False,  # Default causality for MHA within this block
        qkv_bias: bool = True,  # Bias for QKV projections in MHA
        mlp_bias: bool = True,  # Bias for Linear layers in MLP
        use_moe: bool = False, # New: Whether to use MoE instead of MLP
        num_experts: int = 0, # New: Number of experts if use_moe is True
        top_k: int = 0, # New: Number of top experts to select if use_moe is True
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initializes the TransformerBlock.

        Args:
            hidden_size (int): Dimensionality of the input and output.
            num_heads (int): Number of attention heads for MHA.
            mlp_intermediate_size (int, optional): Intermediate size for MLP.
                                                   Defaults to 4 * hidden_size.
            attn_dropout_p (float, default=0.1): Dropout for MHA attention and output.
            mlp_dropout_p (float, default=0.1): Dropout for MLP.
            mlp_activation (str | nn.Module, default="gelu"): Activation for MLP.
            norm_eps (float, default=1e-5): Epsilon for Layer Normalization.
            norm_first (bool, default=True): True for Pre-LN, False for Post-LN.
            is_causal (bool, default=False): Default causality for MHA.
            qkv_bias (bool, default=True): Whether MHA QKV projections should use bias.
            mlp_bias (bool, default=True): Whether MLP Linear layers should use bias.
            use_moe (bool, default=False): Whether to use a Mixture of Experts (MoE) layer instead of a standard MLP.
            num_experts (int, default=0): The total number of experts if `use_moe` is True.
            top_k (int, default=0): The number of top experts to select if `use_moe` is True.
            device (torch.device | str | None, default=None): Target device.
            dtype (torch.dtype | None, default=None): Target data type.
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm_first = norm_first
        self.hidden_size = hidden_size  # Needed for potential checks or logging

        # Initialize Layer Normalization layers
        self.norm1 = nn.LayerNorm(hidden_size, eps=norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, eps=norm_eps, **factory_kwargs)

        # Initialize Multi-Head Attention
        # MHA's internal norm/residual are disabled; TransformerBlock handles them.
        self.self_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            p=attn_dropout_p,
            bias=qkv_bias,  # Pass bias for QKV layers
            is_causal=is_causal,  # Set default causality for MHA
            include_norm_residual=False,  # MHA does not handle norm/residual itself
            eps=norm_eps,  # MHA's norm_eps, not used if include_norm_residual=False
            norm_first=False,  # MHA's norm_first, not used if include_norm_residual=False
            **factory_kwargs,
        )

        # Initialize MLP or MoE
        if mlp_intermediate_size is None:
            mlp_intermediate_size = 4 * hidden_size

        if use_moe:
            if num_experts <= 0 or top_k <= 0:
                raise ValueError("num_experts and top_k must be positive if use_moe is True.")
            self.mlp = MoE(
                hidden_size=hidden_size,
                num_experts=num_experts,
                top_k=top_k,
                intermediate_size=mlp_intermediate_size,
                activation=mlp_activation,
                dropout_p=mlp_dropout_p,
                bias=mlp_bias,
                norm_eps=norm_eps, # Pass norm_eps to MoE's experts
                **factory_kwargs,
            )
        else:
            self.mlp = MLP(
                hidden_size=hidden_size,
                intermediate_size=mlp_intermediate_size,
                activation=mlp_activation,
                dropout_p=mlp_dropout_p,
                bias=mlp_bias,  # Pass bias for MLP layers
                include_norm_residual=False,  # MLP does not handle norm/residual itself
                norm_eps=norm_eps,  # MLP's norm_eps, not used if include_norm_residual=False
                norm_first=False,  # MLP's norm_first, not used if include_norm_residual=False
                **factory_kwargs,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer block.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [B, S, H].
            attn_mask (torch.Tensor, optional): Attention mask for MHA.
            is_causal (bool, optional): Overrides the default MHA causality for this pass.
                                        If None, MHA's default `is_causal` is used.

        Returns:
            torch.Tensor: Output tensor of shape [B, S, H].
        """
        # Determine causality for the MHA call
        # If is_causal is provided as an argument, it overrides the MHA's default.
        # Otherwise, MHA uses its own self.is_causal.
        # The MHA forward method handles this logic if is_causal=None is passed.

        if self.norm_first:  # Pre-LN
            # First sublayer: Multi-Head Attention
            normed_hidden_states = self.norm1(hidden_states)
            attn_output = self.self_attn(
                normed_hidden_states,
                attn_mask=attn_mask,
                is_causal=is_causal,  # Pass through, MHA handles None
            )
            # Residual connection for MHA
            hidden_states = hidden_states + attn_output

            # Second sublayer: MLP
            normed_intermediate_states = self.norm2(hidden_states)
            mlp_output = self.mlp(normed_intermediate_states)
            # Residual connection for MLP
            output = hidden_states + mlp_output

        else:  # Post-LN
            # First sublayer: Multi-Head Attention
            attn_output = self.self_attn(
                hidden_states,
                attn_mask=attn_mask,
                is_causal=is_causal,  # Pass through, MHA handles None
            )
            # Residual connection and normalization for MHA
            hidden_states = self.norm1(hidden_states + attn_output)

            # Second sublayer: MLP
            mlp_output = self.mlp(hidden_states)
            # Residual connection and normalization for MLP
            output = self.norm2(hidden_states + mlp_output)

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
