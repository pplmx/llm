"""Tests for the standalone MultimodalModel / ModalityFusion (TASK-227)."""

from __future__ import annotations

import pytest
import torch

from llm.runtime import ModelFactory
from llm.training.core.config import ModelConfig


def _decoder(use_rope: bool = True):
    return ModelFactory.from_config(
        ModelConfig(vocab_size=32, hidden_size=24, num_layers=2, num_heads=2, max_seq_len=24, use_rope=use_rope)
    )


def test_modality_fusion_projects_to_hidden_and_handles_2d():
    from llm.multimodal.model import ModalityFusion

    fusion = ModalityFusion(modal_dim=16, hidden_size=24)
    # 2D sample-level embedding -> (B, 1, hidden)
    out = fusion(torch.randn(4, 16))
    assert out.shape == (4, 1, 24)
    # 3D prefix embeddings -> (B, M, hidden)
    out3 = fusion(torch.randn(4, 3, 16))
    assert out3.shape == (4, 3, 24)


def test_multimodal_model_returns_text_logits_of_input_length():
    from llm.multimodal.model import MultimodalModel

    decoder = _decoder(use_rope=True)
    model = MultimodalModel(decoder, num_modal_tokens=2, modal_dim=16)
    input_ids = torch.randint(0, 32, (3, 10))
    modal_embeds = torch.randn(3, 2, 16)  # 2 modal prefix tokens
    logits = model(input_ids, modal_embeds)
    assert logits.shape == (3, 10, 32)  # text logits only, prefix excluded


def test_multimodal_model_rejects_invalid_num_modal_tokens():
    from llm.multimodal.model import MultimodalModel

    with pytest.raises(ValueError, match="num_modal_tokens"):
        MultimodalModel(_decoder(), num_modal_tokens=0)


def test_multimodal_model_does_not_patch_decoder():
    """The wrapped DecoderModel stays text-only and patched-free."""
    from llm.multimodal.model import MultimodalModel

    decoder = _decoder(use_rope=True)
    _ = MultimodalModel(decoder, num_modal_tokens=1, modal_dim=16)
    input_ids = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        text_logits = decoder(input_ids)
    assert text_logits.shape == (2, 8, 32)  # unchanged text-only forward
