"""Tests for Decoder norm_impl integration."""

import pytest
import torch

from llm.core.rms_norm import RMSNorm
from llm.models.decoder import DecoderModel


@pytest.mark.slow
def test_decoder_with_rms_norm_impl():
    """Requirement: norm_impl=rms_norm wires RMSNorm through blocks and final norm."""
    vocab_size = 100
    hidden_size = 64
    model = DecoderModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=1,
        num_heads=4,
        norm_impl="rms_norm",
    )
    x = torch.randint(0, vocab_size, (2, 10))
    output = model(x)
    assert output.shape == (2, 10, vocab_size)
    assert type(model.final_norm) is RMSNorm
    assert type(model.transformer_blocks[0].norm1) is RMSNorm
