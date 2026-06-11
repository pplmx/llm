from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from llm.core.embedding import EmbeddingLayer
from llm.core.kv_cache import KVCache
from llm.core.transformer_block import TransformerBlock
from llm.utils.common import make_factory_kwargs


def _resolve_norm_type(norm_impl: str) -> type[nn.Module]:
    from llm.core.registry import NORM_REGISTRY, ensure_norms_registered

    ensure_norms_registered()
    return NORM_REGISTRY.get(norm_impl)


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
        intermediate_size: int | None = None,
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
        num_experts: int = 0,
        top_k: int = 0,
        num_kv_heads: int | None = None,  # For GQA support
        use_glu: bool = False,
        norm_impl: str = "layer_norm",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        attn_impl: str = "mha",
        mlp_impl: str = "mlp",
        gradient_checkpointing: bool = False,
        window_size: int | None = None,
    ):
        """
        Initializes the DecoderModel.
        """
        super().__init__()
        factory_kwargs = make_factory_kwargs(device, dtype)
        resolved_norm_type = _resolve_norm_type(norm_impl)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.norm_first = norm_first  # Store for final norm logic
        self._gradient_checkpointing = gradient_checkpointing

        self.embedding_layer = EmbeddingLayer(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            pos_encoding_learned=pos_encoding_learned,
            dropout_p=embedding_dropout_p,
            padding_idx=padding_idx,
            **factory_kwargs,
        )

        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    attn_dropout_p=attn_dropout_p,
                    mlp_dropout_p=mlp_dropout_p,
                    mlp_activation=mlp_activation,
                    norm_eps=norm_eps,
                    norm_first=norm_first,
                    is_causal=is_causal,  # Pass overall model's causality default
                    qkv_bias=qkv_bias,
                    mlp_bias=mlp_bias,
                    num_experts=num_experts,
                    top_k=top_k,
                    num_kv_heads=num_kv_heads,
                    use_glu=use_glu,  # Pass use_glu
                    norm_type=resolved_norm_type,
                    window_size=window_size,  # Pass window_size
                    attn_impl=attn_impl,  # Pass attn_impl
                    mlp_impl=mlp_impl,  # Pass mlp_impl
                    **factory_kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = None
        if self.norm_first:
            if isinstance(resolved_norm_type, type):
                self.final_norm = resolved_norm_type(hidden_size, eps=norm_eps, **factory_kwargs)
            else:
                self.final_norm = resolved_norm_type

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=lm_head_bias, **factory_kwargs)
        self.max_seq_len = max_seq_len

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        kv_caches: list[KVCache] | None = None,
        use_cache: bool = False,
        position_ids: torch.Tensor | None = None,
        batch_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[KVCache]]:
        """
        Forward pass of the DecoderModel.

        Args:
            input_ids: Input token IDs of shape [B, S].
            attn_mask: Optional attention mask broadcastable to SDPA.
            kv_caches: Pre-allocated KV caches, one per transformer layer.
            use_cache: When True, update ``kv_caches`` in place and return them.
            position_ids: Explicit position IDs of shape [B, S].
            batch_indices: Cache slot indices for continuous batching.

        Returns:
            Logits tensor, or ``(logits, kv_caches)`` when ``use_cache=True``.
        """
        if self._gradient_checkpointing and use_cache:
            raise ValueError("Gradient checkpointing is incompatible with use_cache=True. ")

        if use_cache and kv_caches is None:
            raise ValueError("kv_caches is required when use_cache=True.")

        start_pos = 0
        if kv_caches is not None and kv_caches[0].seq_len > 0:
            start_pos = kv_caches[0].seq_len

        hidden_states = self.embedding_layer(input_ids, start_pos=start_pos, position_ids=position_ids)

        for i, block in enumerate(self.transformer_blocks):
            kv_cache = kv_caches[i] if kv_caches is not None else None

            if self._gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    block,
                    hidden_states,
                    attn_mask,
                    None,
                    None,
                    False,
                    use_reentrant=False,
                )
            else:
                block_outputs = block(
                    hidden_states,
                    attn_mask=attn_mask,
                    is_causal=None,
                    kv_cache=kv_cache,
                    use_cache=use_cache,
                    batch_indices=batch_indices,
                    start_pos=position_ids if (batch_indices is not None and position_ids is not None) else start_pos,
                )
                if use_cache:
                    hidden_states, _current_kv = block_outputs
                else:
                    hidden_states = block_outputs

        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)

        logits = self.lm_head(hidden_states)

        if use_cache:
            return logits, kv_caches
        return logits

    @property
    def gradient_checkpointing(self) -> bool:
        """Whether gradient checkpointing is enabled."""
        return self._gradient_checkpointing

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to reduce memory usage during training."""
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
