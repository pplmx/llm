"""Shared fixtures for DecoderModel tests."""

import pytest
import torch

from llm.models.decoder import DecoderModel
from tests.support.devices import ALL_DEVICES
from tests.support.models import DECODER_BATCH_SIZE, DECODER_SEQ_LEN, decoder_model_kwargs

BATCH_SIZE = DECODER_BATCH_SIZE
SEQ_LEN = DECODER_SEQ_LEN

# Prioritise GPU: use cuda_usable() so visible-but-OOM GPUs are excluded.
DEVICES = ALL_DEVICES


@pytest.fixture
def model_kwargs(request):
    """Default DecoderModel kwargs; override via @pytest.mark.parametrize(..., indirect=True)."""
    kwargs = decoder_model_kwargs()
    if hasattr(request, "param"):
        kwargs.update(request.param)
    return kwargs


@pytest.fixture
def decoder_model(model_kwargs):
    model = DecoderModel(**model_kwargs)
    model.eval()
    return model


@pytest.fixture
def input_ids_tensor(model_kwargs):
    return torch.randint(
        0,
        model_kwargs["vocab_size"],
        (BATCH_SIZE, SEQ_LEN),
        device=model_kwargs["device"],
        dtype=torch.long,
    )


@pytest.fixture
def attention_mask_tensor(model_kwargs):
    mask = torch.zeros(BATCH_SIZE, SEQ_LEN, device=model_kwargs["device"], dtype=torch.bool)
    if SEQ_LEN > 1:
        mask[0, -1] = True
    return mask.unsqueeze(1).unsqueeze(1)
