"""A standalone multimodal decoder model that fuses modal embeddings (TASK-227).

Per ADR-013, this is composed WITHOUT patching :class:`DecoderModel`. A
:class:`MultimodalModel` wraps a plain ``DecoderModel`` and injects the
registry-encoded modal embeddings as a small **prefix** in token-embedding
space: ``fused = [modal_tokens | text_tokens]``, then runs the decoder's
transformer blocks and LM head unchanged and returns the *text* logits.

The modal prefix occupies positions ``0..M-1``; text positions start at ``M``.
RoPE (`use_rope=True`) is the supported positional-encoding path for the prefix
because it is injected inside attention (length-agnostic) rather than an
additive table capped at ``max_seq_len``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from llm.models.decoder import DecoderModel
from llm.multimodal.encoders import ModalityEncoder


class ModalityFusion(nn.Module):
    """Projects modal embeddings into token-embedding space.

    Maps ``(B, *, modal_dim) -> (B, *, hidden_size)`` via a learnable projection
    and applies the same ``sqrt(hidden_size)`` scale the decoder's embedding
    layer uses, so the injected prefix is on the same footing as token
    embeddings.
    """

    def __init__(self, modal_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.modal_dim = int(modal_dim)
        self.hidden_size = int(hidden_size)
        self.proj: nn.Linear
        self.proj = nn.Linear(self.modal_dim, self.hidden_size)

    def forward(self, modal_embeds: torch.Tensor) -> torch.Tensor:
        if modal_embeds.dim() == 2:
            modal_embeds = modal_embeds.unsqueeze(1)  # (B, 1, modal_dim)
        return self.proj(modal_embeds) * math.sqrt(self.hidden_size)


class MultimodalModel(nn.Module):
    """Text decoder conditioned on modality by prepending modal tokens.

    Args:
        decoder: a ``DecoderModel`` (untouched).
        num_modal_tokens: number of prefix tokens per sample (default 1).
        modal_dim: dimension of each sample's registry encoder output; by
            default ``decoder.hidden_size`` (the LinearModalityEncoder default).
        encoder: optional ``ModalityEncoder`` owned by the model so raw
            modality samples can be encoded **in-forward** (trainable vision/
            audio tower, image-text alignment, ROADMAP 12.1/12.2). When None
            the model consumes precomputed ``modal_embeds`` instead (CLIP-style
            frozen-tower path, backward compatible).
    """

    def __init__(
        self,
        decoder: DecoderModel,
        num_modal_tokens: int = 1,
        modal_dim: int | None = None,
        encoder: ModalityEncoder | None = None,
    ) -> None:
        super().__init__()
        if num_modal_tokens < 1:
            raise ValueError(f"num_modal_tokens must be >= 1, got {num_modal_tokens}")
        self.decoder = decoder
        self.num_modal_tokens = int(num_modal_tokens)
        self.hidden_size = decoder.hidden_size
        self.encoder = encoder
        self.fusion = ModalityFusion(modal_dim or self.hidden_size, self.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        modal_embeds: torch.Tensor | None = None,
        modal_samples: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if modal_samples is not None:
            if self.encoder is None:
                raise ValueError("MultimodalModel received modal_samples but has no encoder (construct it with one)")
            modal_embeds = self.encoder.encode(modal_samples)  # (B, M, embed_dim)
        elif modal_embeds is None:
            raise ValueError("MultimodalModel.forward needs modal_embeds or modal_samples")
        text_h = self.decoder.embedding_layer(input_ids)
        modal_h = self.fusion(modal_embeds)  # (B, M, hidden)
        fused = torch.cat([modal_h, text_h], dim=1)  # (B, M+text_len, hidden)
        num_prefix = modal_h.size(1)

        hidden = fused
        for block in self.decoder.transformer_blocks:
            hidden = block(hidden, attn_mask=attn_mask, is_causal=None, use_cache=False)
        if self.decoder.final_norm is not None:
            hidden = self.decoder.final_norm(hidden)
        logits = self.decoder.lm_head(hidden)  # (B, M+text_len, vocab)
        return logits[:, num_prefix:]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        modal_embeds: torch.Tensor | None = None,
        modal_samples: torch.Tensor | None = None,
        max_new_tokens: int = 8,
    ) -> torch.Tensor:
        """Greedy autoregressive decode conditioned on a modal prefix.

        Starts from ``input_ids`` (e.g. an ASR instruction prompt) fused after
        the modal prefix, and appends one argmax token at a time for
        ``max_new_tokens`` steps. Returns ``[B, P + max_new_tokens]`` (prompt
        retained). The whole sequence is re-embedded each step (no KV cache),
        which is fine for the small CPU-verifiable sequences this framework
        targets.
        """
        if modal_embeds is None and modal_samples is None:
            raise ValueError("MultimodalModel.generate needs modal_embeds or modal_samples")
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if modal_embeds is None:
            if modal_samples is None or self.encoder is None:
                raise ValueError("MultimodalModel.generate needs modal_embeds or a raw modal sample with an encoder")
            modal_embeds = self.encoder.encode(modal_samples.to(device))
        if modal_embeds.dim() == 2:
            modal_embeds = modal_embeds.unsqueeze(1)
        modal_embeds = modal_embeds.to(device)

        # A zero-length prompt (e.g. pure audio->text ASR with no instruction)
        # is seeded with a neutral token and that seed is stripped afterwards,
        # so generation still needs at least one sampled token.
        empty_prompt = input_ids.shape[-1] == 0
        if empty_prompt:
            seed = torch.full((input_ids.shape[0], 1), 1, dtype=torch.long, device=device)
            fused = seed
        else:
            fused = input_ids
        for _ in range(int(max_new_tokens)):
            logits = self(fused, modal_embeds=modal_embeds)  # (B, cur_len, V)
            next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)  # (B, 1)
            fused = torch.cat([fused, next_tok], dim=1)
        if empty_prompt:
            return fused[:, 1:]
        return fused


def build_multimodal_model(decoder: DecoderModel, num_modal_tokens: int = 1, modal_dim: int | None = None):
    return MultimodalModel(decoder, num_modal_tokens=num_modal_tokens, modal_dim=modal_dim)
