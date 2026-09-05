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
    model = MultimodalModel(decoder, modal_dim=16)
    input_ids = torch.randint(0, 32, (3, 10))
    modal_embeds = torch.randn(3, 2, 16)  # 2 modal prefix tokens
    logits = model(input_ids, modal_embeds)
    assert logits.shape == (3, 10, 32)  # text logits only, prefix excluded


def test_multimodal_model_prefix_width_is_encoder_driven():
    """Regression (RIL TASK-311/ISS-349): the old ``num_modal_tokens``
    constructor knob was stored but never read — forward/generate derive the
    prefix width from ``modal_embeds`` itself, so a 'knob' value could not
    take effect (setting 1 vs 9 produced bit-identical output). The knob is
    removed; the prefix width follows the encoder's ``num_tokens`` and the
    model simply slices off whatever the encoder emitted."""
    from llm.multimodal.model import MultimodalModel

    decoder = _decoder(use_rope=True)
    model = MultimodalModel(decoder, modal_dim=16)
    input_ids = torch.randint(0, 32, (2, 8))
    for n_prefix in (1, 4):
        modal_embeds = torch.randn(2, n_prefix, 16)
        logits = model(input_ids, modal_embeds)
        assert logits.shape == (2, 8, 32)  # text logits only, prefix stripped
    with pytest.raises(TypeError):
        # The constructor no longer accepts the dead knob.
        MultimodalModel(decoder, num_modal_tokens=1)


def test_multimodal_model_does_not_patch_decoder():
    """The wrapped DecoderModel stays text-only and patched-free."""
    from llm.multimodal.model import MultimodalModel

    decoder = _decoder(use_rope=True)
    _ = MultimodalModel(decoder, modal_dim=16)
    input_ids = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        text_logits = decoder(input_ids)
    assert text_logits.shape == (2, 8, 32)  # unchanged text-only forward
