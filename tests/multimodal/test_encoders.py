"""Tests for the modality-encoder registry + minimal CPU encoder (TASK-226)."""

from __future__ import annotations

import pytest
import torch


def test_registry_has_linear_encoder():
    from llm.multimodal import MODALITY_ENCODER_REGISTRY

    assert "linear" in MODALITY_ENCODER_REGISTRY
    cls = MODALITY_ENCODER_REGISTRY.get("linear")
    enc = cls(16, 24)
    assert enc.input_dim == 16
    assert enc.embed_dim == 24


def test_linear_encoder_projects_to_embed_dim():
    from llm.multimodal import MODALITY_ENCODER_REGISTRY

    enc = MODALITY_ENCODER_REGISTRY.get("linear")(16, 24)
    x = torch.randn(4, 16)
    out = enc(x)
    assert out.shape == (4, 24)
    assert torch.isfinite(out).all()


def test_linear_encoder_trains():
    from llm.multimodal import MODALITY_ENCODER_REGISTRY

    enc = MODALITY_ENCODER_REGISTRY.get("linear")(8, 8)
    x = torch.randn(2, 8)
    loss = enc(x).sum()
    loss.backward()
    assert enc.proj.weight.grad is not None
    assert enc.proj.weight.grad.abs().sum() > 0


def test_base_encoder_is_abstract():
    from llm.multimodal.encoders import ModalityEncoder

    with pytest.raises(TypeError):
        ModalityEncoder()  # cannot instantiate the abstract base


def test_duplicate_registration_raises():
    from llm.multimodal.encoders import LinearModalityEncoder, register_encoder

    with pytest.raises(ValueError, match="already registered"):
        register_encoder("linear")(LinearModalityEncoder)
