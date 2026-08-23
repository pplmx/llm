"""Image -> token preprocessor (ROADMAP 12.3 / multimodal research slice).

The multimodal spike (``multimodal/encoders.py`` + ``data.py``) uses a plain
linear encoder as a stand-in for a real vision encoder, and consumes raw feature
vectors — there is no real *image-to-token* step. This slice provides that
missing piece: a CPU-verifiable ViT-style preprocessor which

1. slices an ``[B, C, H, W]`` image into ``(H/p)·(W/p)`` square patches
   (:func:`patchify`),
2. flattens each patch and projects it to ``embed_dim`` via a learned linear
   projection (``nn.Linear``),
3. adds a learnable positional embedding (like ViT) so position information
   enters the token stream.

The output is ``[B, num_patches, embed_dim]`` image-token embeddings, the
standard format a multimodal backbone consumes. Self-contained and CPU-
verifiable; wiring it (plus a real CLIP/SigLIP encoder) into the multimodal
model is a documented follow-up — it does not patch :class:`DecoderModel`.

CPU parity invariants verified in ``tests/multimodal/test_preprocess.py``:

- :func:`patchify` equals an explicit manual patch-slicing reference;
- the output shape is ``[B, num_patches, embed_dim]`` and deterministic;
- gradients flow to the projection / positional embedding parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["ImagePatchPreprocessor", "patchify"]


def patchify(image: Tensor, patch_size: int) -> Tensor:
    """Split ``[B, C, H, W]`` into ``[B, num_patches, C*p*p]`` patches.

    Requires ``H`` and ``W`` divisible by ``patch_size``.
    """
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    batch, channels, height, width = image.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"image ({height}x{width}) not divisible by patch_size={patch_size}")
    # [B, C, H/p, W/p, p, p] -> [B, num_patches, C*p*p]
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    n_h, n_w = height // patch_size, width // patch_size
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch, n_h * n_w, channels * patch_size * patch_size)
    return patches


class ImagePatchPreprocessor(nn.Module):
    """Image-to-token preprocessor: patchify -> linear projection + pos embedding.

    Args:
        in_channels: Image channels (e.g. 3 for RGB).
        patch_size: Square patch side length.
        embed_dim: Projected token embedding dimension.
        max_patches: Maximum number of patches for the positional embedding;
            defaults to ``(image_h // patch_size) * (image_w // patch_size)``
            supplied at construction via ``image_h``/``image_w``.
        image_h/image_w: Reference spatial size used only to size the positional
            embedding (truncate/raise if a forward image has more patches).
    """

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        image_h: int = 224,
        image_w: int = 224,
    ) -> None:
        super().__init__()
        if image_h % patch_size != 0 or image_w % patch_size != 0:
            raise ValueError(f"image {image_h}x{image_w} not divisible by patch_size={patch_size}")
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim, bias=False)
        num_patches = (image_h // patch_size) * (image_w // patch_size)
        # Learnable positional embedding (ViT-style), [1, num_patches, embed_dim].
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, image: Tensor) -> Tensor:
        """``[B, C, H, W]`` -> ``[B, num_patches, embed_dim]`` image tokens."""
        num_patches = self.pos_embed.shape[1]
        patches = patchify(image, self.patch_size)  # [B, N, C*p*p]
        if patches.shape[1] != num_patches:
            raise ValueError(
                f"image produced {patches.shape[1]} patches but positional embedding has {num_patches} slots"
            )
        tokens = self.proj(patches)  # [B, N, D]
        return tokens + self.pos_embed
