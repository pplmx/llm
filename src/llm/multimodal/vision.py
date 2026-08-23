"""CLIP/SigLIP-style vision transformer encoder (ROADMAP 12.1 slice 1).

A real (CPU-verifiable) vision encoder registered in
:data:`MODALITY_ENCODER_REGISTRY` as ``"vit"``. Unlike the placeholder
:class:`~llm.multimodal.encoders.LinearModalityEncoder` — which consumes
*precomputed feature vectors* — this encoder consumes **raw images**
``[B, C, H, W]`` and emits image-token embeddings in the standard ViT layout
that :class:`~llm.multimodal.model.MultimodalModel` fuses as a text prefix.

Architecture (ViT-B-style tower):

1. :class:`~llm.multimodal.preprocess.ImagePatchPreprocessor` slices each image
   into ``N=(H/p)*(W/p)`` square patches, projects each patch to ``embed_dim``
   (learned linear) and adds a learnable positional embedding (ROADMAP 12.3);
2. an optional learnable ``[CLS]`` token row is prepended (CLIP-style);
3. ``layers`` pre-norm transformer blocks run over the sequence: LayerNorm ->
   multi-head scaled-dot-product attention -> residual -> LayerNorm -> MLP
   (GELU, ``mlp_ratio`` expansion) -> residual;
4. a final LayerNorm (SigLIP/CLIP penultimate-normalization style).

Output is ``[B, N (+1 if CLS/with_cls), embed_dim]`` image-token embeddings. The
tower is standalone — it does not touch :class:`DecoderModel` (ADR-013) — and is
fully CPU-verifiable. ``freeze_encoder=True`` replicates the common CLIP-style
practice of keeping the vision tower fixed while downstream heads train.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from llm.multimodal.encoders import ModalityEncoder, register_encoder
from llm.multimodal.preprocess import ImagePatchPreprocessor

__all__ = ["VisionTransformerBlock", "VisionTransformerEncoder"]


class VisionTransformerBlock(nn.Module):
    """Pre-norm ViT encoder block (LayerNorm -> MHA -> residual -> MLP -> residual).

    Mirrors the residual structure of the decoder's ``TransformerBlock``
    (``norm_first=True``) so the two towers compose naturally, but is a
    standalone implementation: patch context is bidirectional (no causal mask)
    and position comes from the learned positional embedding (no RoPE).
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout_p: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} not divisible by num_heads={num_heads}")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.q = nn.Linear(self.embed_dim, self.embed_dim)
        self.k = nn.Linear(self.embed_dim, self.embed_dim)
        self.v = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn_drop_p = float(dropout_p)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        hidden = int(self.embed_dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, self.embed_dim)
        self.mlp_drop_p = float(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm1(x)
        batch, length, dim = h.shape
        q = self.q(h).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        attn = functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )
        x = residual + self.out_proj(attn.transpose(1, 2).reshape(batch, length, dim))

        residual = x
        h = self.norm2(x)
        x = residual + functional.dropout(self.fc2(self.act(self.fc1(h))), p=self.mlp_drop_p, training=self.training)
        return x


@register_encoder("vit")
class VisionTransformerEncoder(ModalityEncoder, nn.Module):
    """CLIP/SigLIP-style ViT image encoder registered as ``"vit"``.

    Args:
        in_channels: Image channels (e.g. 3 for RGB).
        patch_size: Square patch side length.
        embed_dim: Token embedding dimension (function of ``num_heads``).
        layers: Number of transformer blocks.
        num_heads: Attention heads per block (must divide ``embed_dim``).
        mlp_ratio: MLP hidden expansion factor.
        image_h/image_w: Reference spatial size; also sizes the patch count of
            the positional embedding (``image_h``/``image_w`` > 0 enforced by
            :class:`ImagePatchPreprocessor`).
        with_cls: Prepend a learnable ``[CLS]`` token (CLIP-style) so the first
            output row pools the whole image.
        freeze_encoder: Set ``requires_grad=False`` on every parameter
            (frozen vision tower, the common CLIP fine-tuning setup).
        dropout_p: Dropout probability on attention and MLP (training only).
    """

    modality = "vit"

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        image_h: int = 224,
        image_w: int = 224,
        with_cls: bool = True,
        freeze_encoder: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {embed_dim}")
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.input_dim = self.in_channels * self.patch_size * self.patch_size
        self.image_h = int(image_h)
        self.image_w = int(image_w)
        self.num_patches = (self.image_h // patch_size) * (self.image_w // patch_size)

        self.patch_embed = ImagePatchPreprocessor(
            in_channels=self.in_channels,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            image_h=self.image_h,
            image_w=self.image_w,
        )
        if with_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None  # type: ignore[assignment]
        self.blocks = nn.ModuleList(
            [VisionTransformerBlock(self.embed_dim, self.num_heads, mlp_ratio, dropout_p) for _ in range(int(layers))]
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        if freeze_encoder:
            self._freeze()

    def _freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    @property
    def num_tokens(self) -> int:
        """Sequence length this encoder emits per image (N, or N+1 with ``[CLS]``)."""
        return self.num_patches + (1 if self.cls_token is not None else 0)

    def encode(self, sample: torch.Tensor) -> torch.Tensor:
        """``[B, C, H, W]`` -> ``[B, num_tokens, embed_dim]`` image-token embeddings."""
        tokens = self.patch_embed(sample)  # [B, N, D]
        if self.cls_token is not None:
            cls = self.cls_token.expand(sample.shape[0], -1, -1)  # [B, 1, D]
            tokens = torch.cat([cls, tokens], dim=1)  # [B, N+1, D]
        hidden = tokens
        for block in self.blocks:
            hidden = block(hidden)
        return self.norm(hidden)
