import torch
import torch.nn as nn

from llm.core.embedding import EmbeddingLayer
from llm.core.transformer_block import TransformerBlock


class DecoderModel(nn.Module):
    """
    A Transformer-based decoder model.

    This model consists of an embedding layer, a stack of Transformer blocks,
    an optional final layer normalization (for Pre-LN architectures), and a
    language modeling head to predict token logits.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 512,
        mlp_intermediate_size: int | None = None,
        pos_encoding_learned: bool = False,
        embedding_dropout_p: float = 0.1,
        attn_dropout_p: float = 0.1,
        mlp_dropout_p: float = 0.1,
        mlp_activation: str | nn.Module = "gelu",
        norm_eps: float = 1e-5,
        norm_first: bool = True,
        is_causal: bool = True,  # Default to True for a decoder model
        padding_idx: int | None = None,
        qkv_bias: bool = True,  # Bias for QKV in MHA within TransformerBlock
        mlp_bias: bool = True,  # Bias for MLP in TransformerBlock
        lm_head_bias: bool = True,  # Bias for the final LM head
        use_moe: bool = False, # New: Whether to use MoE in TransformerBlocks
        num_experts: int = 0, # New: Number of experts if use_moe is True
        top_k: int = 0, # New: Number of top experts to select if use_moe is True
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initializes the DecoderModel.

        Args:
            vocab_size (int): Vocabulary size.
            hidden_size (int): Dimensionality of the model.
            num_layers (int): Number of TransformerBlock layers.
            num_heads (int): Number of attention heads per TransformerBlock.
            max_seq_len (int, default=512): Max sequence length for embeddings.
            mlp_intermediate_size (int, optional): Intermediate size for MLPs in blocks.
                                                   Defaults to 4 * hidden_size.
            pos_encoding_learned (bool, default=False): Use learned positional embeddings.
            embedding_dropout_p (float, default=0.1): Dropout for positional encoding.
            attn_dropout_p (float, default=0.1): Dropout for MHA in blocks.
            mlp_dropout_p (float, default=0.1): Dropout for MLP in blocks.
            mlp_activation (str | nn.Module, default="gelu"): Activation for MLP in blocks.
            norm_eps (float, default=1e-5): Epsilon for LayerNorms.
            norm_first (bool, default=True): True for Pre-LN, False for Post-LN.
            is_causal (bool, default=True): If MHA in blocks should be causal.
            padding_idx (int, optional, default=None): Padding index for embeddings.
            qkv_bias (bool, default=True): Bias for QKV projections in MHA.
            mlp_bias (bool, default=True): Bias for Linear layers in MLP.
            lm_head_bias (bool, default=True): Bias for the final language modeling head.
            use_moe (bool, default=False): Whether to use a Mixture of Experts (MoE) layer in TransformerBlocks.
            num_experts (int, default=0): The total number of experts if `use_moe` is True.
            top_k (int, default=0): The number of top experts to select if `use_moe` is True.
            device (torch.device | str | None, default=None): Target device.
            dtype (torch.dtype | None, default=None): Target data type.
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.norm_first = norm_first  # Store for final norm logic

        self.embedding_layer = EmbeddingLayer(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            pos_encoding_learned=pos_encoding_learned,
            dropout_p=embedding_dropout_p,
            padding_idx=padding_idx,
            **factory_kwargs,
        )

        if mlp_intermediate_size is None:
            mlp_intermediate_size = 4 * hidden_size

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_intermediate_size=mlp_intermediate_size,
                    attn_dropout_p=attn_dropout_p,
                    mlp_dropout_p=mlp_dropout_p,
                    mlp_activation=mlp_activation,
                    norm_eps=norm_eps,
                    norm_first=norm_first,
                    is_causal=is_causal,  # Pass overall model's causality default
                    qkv_bias=qkv_bias,
                    mlp_bias=mlp_bias,
                    use_moe=use_moe, # Pass MoE flag
                    num_experts=num_experts, # Pass num_experts
                    top_k=top_k, # Pass top_k
                    **factory_kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = None
        if self.norm_first:
            self.final_norm = nn.LayerNorm(hidden_size, eps=norm_eps, **factory_kwargs)

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=lm_head_bias, **factory_kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        # is_causal for individual forward passes is not typically exposed at this level,
        # as the decoder model's causality is a structural property set at init.
        # If a block needs dynamic causality, it would be an argument to the block's forward.
        # Here, attn_mask is the primary way to influence attention beyond the default causality.
    ) -> torch.Tensor:
        """
        Forward pass of the DecoderModel.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape [B, S].
            attn_mask (torch.Tensor, optional): Attention mask, typically a padding mask.
                Shape should be broadcastable to [B, N, S, S] or compatible with
                `torch.nn.functional.scaled_dot_product_attention`.
                If `is_causal=True` in blocks, this mask will be combined with the causal mask.

        Returns:
            torch.Tensor: Logits tensor of shape [B, S, vocab_size].
        """
        hidden_states = self.embedding_layer(input_ids)

        for block in self.transformer_blocks:
            # The `is_causal` parameter in block.forward() can override the block's
            # default. Here, we rely on the block's initialized `is_causal` state.
            # So, we pass `is_causal=None` to the block's forward method.
            hidden_states = block(hidden_states, attn_mask=attn_mask, is_causal=None)

        if self.final_norm is not None:  # Applied only in Pre-LN architectures
            hidden_states = self.final_norm(hidden_states)

        logits = self.lm_head(hidden_states)
        return logits


if __name__ == "__main__":
    # Example Usage
    device_ex = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_ex = torch.float32

    vocab_size_ex = 1000
    hidden_size_ex = 64
    num_layers_ex = 2
    num_heads_ex = 4
    max_seq_len_ex = 128
    batch_size_ex = 4
    seq_len_ex = 50  # Test sequence length

    print(f"DecoderModel Example running on device: {device_ex}, dtype: {dtype_ex}")

    # Dummy input
    dummy_input_ids = torch.randint(0, vocab_size_ex, (batch_size_ex, seq_len_ex), device=device_ex, dtype=torch.long)
    # Example padding mask (True means ignore): last 10 tokens are padding for first batch item
    example_padding_mask = torch.zeros(
        batch_size_ex,
        1,
        1,
        seq_len_ex,  # MHA expects [B, N, S_q, S_k] or broadcastable
        device=device_ex,
        dtype=torch.bool,
    )
    if seq_len_ex > 10:
        example_padding_mask[0, 0, 0, -10:] = True

    # --- Test Pre-LN Decoder ---
    print("\nTesting Pre-LN DecoderModel (is_causal=True by default)...")
    pre_ln_decoder = DecoderModel(
        vocab_size=vocab_size_ex,
        hidden_size=hidden_size_ex,
        num_layers=num_layers_ex,
        num_heads=num_heads_ex,
        max_seq_len=max_seq_len_ex,
        norm_first=True,
        device=device_ex,
        dtype=dtype_ex,
    )
    pre_ln_decoder.eval()

    output_pre_ln = pre_ln_decoder(dummy_input_ids)
    print(f"  Input IDs shape: {dummy_input_ids.shape}")
    print(f"  Output logits shape (Pre-LN): {output_pre_ln.shape}")
    assert output_pre_ln.shape == (batch_size_ex, seq_len_ex, vocab_size_ex)
    assert pre_ln_decoder.final_norm is not None

    print("\nTesting Pre-LN DecoderModel with padding mask...")
    output_pre_ln_masked = pre_ln_decoder(dummy_input_ids, attn_mask=example_padding_mask)
    print(f"  Output logits shape (Pre-LN, masked): {output_pre_ln_masked.shape}")
    assert output_pre_ln_masked.shape == (batch_size_ex, seq_len_ex, vocab_size_ex)

    # --- Test Post-LN Decoder ---
    print("\nTesting Post-LN DecoderModel (is_causal=True by default)...")
    post_ln_decoder = DecoderModel(
        vocab_size=vocab_size_ex,
        hidden_size=hidden_size_ex,
        num_layers=num_layers_ex,
        num_heads=num_heads_ex,
        max_seq_len=max_seq_len_ex,
        norm_first=False,  # Key change for Post-LN
        device=device_ex,
        dtype=dtype_ex,
    )
    post_ln_decoder.eval()

    output_post_ln = post_ln_decoder(dummy_input_ids)
    print(f"  Input IDs shape: {dummy_input_ids.shape}")
    print(f"  Output logits shape (Post-LN): {output_post_ln.shape}")
    assert output_post_ln.shape == (batch_size_ex, seq_len_ex, vocab_size_ex)
    assert post_ln_decoder.final_norm is None

    print("\nTesting Post-LN DecoderModel with padding mask...")
    output_post_ln_masked = post_ln_decoder(dummy_input_ids, attn_mask=example_padding_mask)
    print(f"  Output logits shape (Post-LN, masked): {output_post_ln_masked.shape}")
    assert output_post_ln_masked.shape == (batch_size_ex, seq_len_ex, vocab_size_ex)

    print("\nAll basic __main__ tests passed.")
