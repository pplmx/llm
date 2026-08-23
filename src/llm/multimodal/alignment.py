"""CLIP/SigLIP-style contrastive image-text alignment (ROADMAP 12.1 slice 3).

The image-text alignment module: a trainable :class:`ContrastiveAligner` that
scores a batch of ``(image, text)`` pairs so matched pairs out-score unmatched
ones via a temperature-scaled contrastive objective — the standard CLIP loss,
optionally in its SigLIP (sigmoid) variant.

It consumes *embeddings already produced by the towers* — e.g. the image-token
embeddings ``[B, N, D]`` from :class:`~llm.multimodal.vision.VisionTransformerEncoder`
(pooled to a single vector) and text hidden states ``[B, T, D]`` from a
``DecoderModel`` — so it composes with :mod:`llm.multimodal` (including the
trainable-tower path) without touching ``DecoderModel``.

The sampled-batch objective is

    logits[i][j] = exp(logit_scale) * <L2(proj_img_i), L2(proj_txt_j)>

with ``loss = CE(logits, labels=arange(B))`` averaged over both directions
(image->text and text->image). CPU-verifiable: with random paired data the
loss starts near ``log(B)`` and a few updates drive retrieval accuracy to 1.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional

__all__ = ["ContrastiveAligner", "contrastive_loss"]


def contrastive_loss(logits: torch.Tensor, *, sigmoid: bool = False) -> torch.Tensor:
    """Symmetric InfoNCE (labels = batch diagonal) or SigLIP binary loss.

    Args:
        logits: square ``[B, B]`` temperature-scaled similarity scores.
        sigmoid: use the SigLIP ``log_sigmoid`` sum over the ``(i, j)`` grid
            (with the diagonal positive) instead of two-direction CE.
    """
    if logits.dim() != 2 or logits.size(0) != logits.size(1):
        raise ValueError(f"logits must be a square [B, B] matrix, got {tuple(logits.shape)}")
    batch = logits.size(0)
    if sigmoid:
        device = logits.device
        eye = torch.eye(batch, dtype=torch.bool, device=device)
        y = torch.where(eye, torch.ones_like(logits), -torch.ones_like(logits))
        return -functional.logsigmoid(y * logits).mean()
    labels = torch.arange(batch, device=logits.device)
    i2t = functional.cross_entropy(logits, labels)
    t2i = functional.cross_entropy(logits.t(), labels)
    return (i2t + t2i) / 2


class ContrastiveAligner(nn.Module):
    """Temperature-scaled contrastive image-text alignment head.

    Projects the pooled image / text embedding into a shared space with
    separate linear heads, L2-normalizes, and scores with a learnable
    temperature (``scale = exp(logit_scale)``, CLIP initialises it at
    ``ln(1/0.07)``).

    Args:
        image_dim: vision embedding dimension (e.g. ``embed_dim`` of
            :class:`~llm.multimodal.vision.VisionTransformerEncoder`).
        text_dim: text embedding dimension (e.g. ``DecoderModel.hidden_size``).
        proj_dim: shared projection dimension of the normalized space.
        sigmoid: score with the SigLIP sigmoid loss instead of symmetric CE.
        image_pool: ``"mean"`` pools all image tokens; ``"cls"`` takes the
            first row (the ``[CLS]`` token row with ``with_cls=True``).
        init_logit_scale: initial log-temperature (CLIP: ``ln(1/0.07)``).
    """

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        proj_dim: int = 256,
        *,
        sigmoid: bool = False,
        image_pool: str = "mean",
        init_logit_scale: float = math.log(1.0 / 0.07),
    ) -> None:
        super().__init__()
        if image_dim <= 0 or text_dim <= 0 or proj_dim <= 0:
            raise ValueError("image_dim, text_dim and proj_dim must all be > 0")
        if image_pool not in ("mean", "cls"):
            raise ValueError(f"image_pool must be 'mean' or 'cls', got {image_pool!r}")
        self.proj_dim = int(proj_dim)
        self.sigmoid = bool(sigmoid)
        self.image_pool = image_pool
        self.image_proj = nn.Linear(int(image_dim), self.proj_dim)
        self.text_proj = nn.Linear(int(text_dim), self.proj_dim)
        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))
        nn.init.kaiming_uniform_(self.image_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.text_proj.weight, a=math.sqrt(5))

    def _pool(self, embeds: torch.Tensor, *, image: bool) -> torch.Tensor:
        """``[B, N, D]`` -> ``[B, D]`` (mean over tokens or the ``[CLS]`` row)."""
        if embeds.dim() == 2:
            return embeds
        if image and self.image_pool == "cls":
            return embeds[:, 0]
        return embeds.mean(dim=1)

    def _normalized(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img = functional.normalize(self.image_proj(self._pool(image_embeds, image=True)), dim=-1)
        txt = functional.normalize(self.text_proj(self._pool(text_embeds, image=False)), dim=-1)
        return img, txt

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """Return ``[B, B]`` temperature-scaled image-text similarity logits."""
        img, txt = self._normalized(image_embeds, text_embeds)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * (img @ txt.t())

    def loss(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """Contrastive loss for the aligned batch (pairs on the diagonal)."""
        return contrastive_loss(self.forward(image_embeds, text_embeds), sigmoid=self.sigmoid)
