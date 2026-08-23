"""Modality encoder registry + minimal CPU encoder (ROADMAP 阶段十二 spike).

This is the *contract first* slice of multimodal support (RIL TASK-226 /
DEC-058). It lands the registration surface and one tiny, CPU-verifiable
encoder; real vision/audio encoders (CLIP/SigLIP etc.) plug in later by
registering under :data:`MODALITY_ENCODER_REGISTRY`. Per the ADR boundary, no
code here patches :class:`llm.models.decoder.DecoderModel` — a multimodal model
consumes encoder outputs via its own new structure in a later slice.
"""

from __future__ import annotations

import abc

import torch
import torch.nn as nn

from llm.runtime.registry import Registry, decorator_register


class ModalityEncoder(abc.ABC, nn.Module):
    """Converts one sample of an auxiliary modality into an embedding tensor.

    Subclasses declare a ``modality`` name and implement :meth:`encode`, which
    turns a raw modal sample (e.g. a feature vector) into a tensor that a
    future multimodal model can consume.
    """

    modality: str = "base"
    input_dim: int
    embed_dim: int

    @abc.abstractmethod
    def encode(self, sample: torch.Tensor) -> torch.Tensor:
        """Encode ``sample`` into a fixed ``embed_dim`` embedding."""

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.encode(sample)


MODALITY_ENCODER_REGISTRY: Registry[type] = Registry("ModalityEncoder")
#: ``@register_encoder("name")`` class decorator (mirrors other registries).
register_encoder = decorator_register(MODALITY_ENCODER_REGISTRY)


@register_encoder("linear")
class LinearModalityEncoder(ModalityEncoder):
    """Minimal learnable encoder: projects a raw feature vector to ``embed_dim``.

    ``encode`` returns ``(sample @ W.T) + b`` via an ``nn.Linear``. It has no
    external image/audio dependencies, so it is fully verifiable on CPU and the
    parameters accept gradients — a faithful stand-in for a real encoder.
    """

    modality = "linear"

    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or embed_dim <= 0:
            raise ValueError("input_dim and embed_dim must be > 0")
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.proj: nn.Linear
        self.proj = nn.Linear(self.input_dim, self.embed_dim)

    def encode(self, sample: torch.Tensor) -> torch.Tensor:
        return self.proj(sample)
