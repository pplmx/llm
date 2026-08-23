"""Whisper-style audio encoder (ROADMAP 12.2 slice 1).

Consumes raw log-mel spectrograms ``[B, 1, T, F]`` (Whisper's input layout, one
channel, ``T`` time frames x ``F`` mel bins) and emits audio-token embeddings
``[B, N (+1 CLS), embed_dim]`` using the same machinery as the vision tower:

1. :class:`~llm.multimodal.preprocess.ImagePatchPreprocessor` slices the 2D
   spectrogram into ``N=(T/p)*(F/p)`` square patches (time x freq), projects
   each to ``embed_dim`` and adds a learnable positional embedding;
2. an optional ``[CLS]`` token is prepended;
3. ``layers`` pre-norm transformer blocks run over the token sequence;
4. a final LayerNorm.

Registered as ``"audio"`` in :data:`MODALITY_ENCODER_REGISTRY` and wired into
:class:`~llm.multimodal.data.MultimodalDataModule` so audio-conditioned text
training runs on CPU (the same raw-sample path the ``"vit"`` encoder uses).
"""

# TODO(multimodal, 12.2): Whisper's real front-end is 2 conv layers + GELU
# over the spectrogram; the patch stem here is the research-framework
# equivalent (conv stem left as a documented variant).
from __future__ import annotations

import torch
import torch.nn as nn

from llm.multimodal.encoders import ModalityEncoder, register_encoder
from llm.multimodal.preprocess import ImagePatchPreprocessor
from llm.multimodal.vision import VisionTransformerBlock

__all__ = ["AudioSpectrogramEncoder"]


@register_encoder("audio")
class AudioSpectrogramEncoder(ModalityEncoder, nn.Module):
    """Whisper-style spectrogram-patch audio encoder registered as ``"audio"``.

    Args:
        patch_size: square patch side over time/freq (both must divide).
        embed_dim: token embedding dimension.
        layers: number of transformer blocks.
        num_heads: attention heads (must divide ``embed_dim``).
        mlp_ratio: MLP hidden expansion factor.
        n_mels: mel-bins per frame (spectrogram width ``F``).
        n_frames: time frames (spectrogram height ``T``).
        with_cls: prepend a learnable ``[CLS]`` token.
        freeze_encoder: freeze all parameters (frozen audio tower).
        dropout_p: attention/MLP dropout (training only).
    """

    modality = "audio"

    def __init__(
        self,
        patch_size: int = 4,
        embed_dim: int = 256,
        layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        n_mels: int = 64,
        n_frames: int = 64,
        with_cls: bool = True,
        freeze_encoder: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {embed_dim}")
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.n_mels = int(n_mels)
        self.n_frames = int(n_frames)
        self.input_dim = 1 * self.patch_size * self.patch_size
        self.num_patches = (self.n_frames // patch_size) * (self.n_mels // patch_size)

        self.patch_embed = ImagePatchPreprocessor(
            in_channels=1,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            image_h=self.n_frames,
            image_w=self.n_mels,
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
            for param in self.parameters():
                param.requires_grad = False

    @property
    def num_tokens(self) -> int:
        """Sequence length this encoder emits per spectrogram (N or N+1 CLS)."""
        return self.num_patches + (1 if self.cls_token is not None else 0)

    def encode(self, sample: torch.Tensor) -> torch.Tensor:
        """``[B, 1, T, F]`` -> ``[B, num_tokens, embed_dim]`` audio-token embeddings."""
        tokens = self.patch_embed(sample)  # [B, N, D]
        if self.cls_token is not None:
            cls = self.cls_token.expand(sample.shape[0], -1, -1)  # [B, 1, D]
            tokens = torch.cat([cls, tokens], dim=1)  # [B, N+1, D]
        hidden = tokens
        for block in self.blocks:
            hidden = block(hidden)
        return self.norm(hidden)
