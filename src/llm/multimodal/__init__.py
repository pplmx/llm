"""Multimodal extension (ROADMAP 阶段十二, RIL TASK-226/TASK-227).

Layer 1 (TASK-226) lands the registry + DataModule contract; Layer 2
(TASK-227) adds a standalone :class:`MultimodalModel` that fuses modal
embeddings into token-embedding space plus a ``multimodal`` training task.
Nothing here patches :class:`DecoderModel` (see
``docs/adr/013-multimodal-encoder-contract.md``).
"""

from __future__ import annotations

from llm.multimodal.data import MultimodalDataModule
from llm.multimodal.encoders import (
    MODALITY_ENCODER_REGISTRY,
    LinearModalityEncoder,
    ModalityEncoder,
    register_encoder,
)
from llm.multimodal.model import ModalityFusion, MultimodalModel, build_multimodal_model
from llm.multimodal.task import MultimodalTask
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
    "ModalityFusion",
    "MultimodalDataModule",
    "MultimodalModel",
    "MultimodalTask",
    "build_multimodal_model",
    "ensure_multimodal_encoders_registered",
    "register_encoder",
]
