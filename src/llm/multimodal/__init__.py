"""Multimodal extension — contract-first spike (ROADMAP 阶段十二, RIL TASK-226).

Establishes the registry + DataModule contract for multimodal data. Real
vision/audio encoders and any model integration are separate future slices;
this module only owns the extensible surface (a modality-encoder registry and
a text+modality ``MultimodalDataModule``). See ``docs/adr/013-multimodal-encoder-contract.md``.
"""

from __future__ import annotations

from llm.multimodal.data import MultimodalDataModule
from llm.multimodal.encoders import (
    MODALITY_ENCODER_REGISTRY,
    LinearModalityEncoder,
    ModalityEncoder,
    register_encoder,
)
from llm.runtime.plugins import load_entry_point_registry


def ensure_multimodal_encoders_registered() -> None:
    """Register built-ins (idempotent) and discover third-party encoders."""
    import llm.multimodal.encoders as _encoders  # noqa: F401 - registers built-ins on import

    load_entry_point_registry("llm.modality_encoders", MODALITY_ENCODER_REGISTRY)


ensure_multimodal_encoders_registered()

__all__ = [
    "MODALITY_ENCODER_REGISTRY",
    "LinearModalityEncoder",
    "ModalityEncoder",
    "MultimodalDataModule",
    "ensure_multimodal_encoders_registered",
    "register_encoder",
]
