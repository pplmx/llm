"""Tests for serving checkpoint loader."""

import string

import pytest
import torch

from llm.serving.config import ServingConfig
from llm.serving.loader import (
    infer_num_layers,
    infer_vocab_size,
    load_model_and_tokenizer,
    load_training_checkpoint,
)
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.training.distributed import model_state_dict


@pytest.fixture
def device():
    """Force CPU for these tests — the session-scoped device fixture from
    conftest.py creates models on CUDA, which OOMs on constrained boxes."""
    return torch.device("cpu")


def test_infer_vocab_size_from_lm_head(tiny_model, tiny_config):
    state = model_state_dict(tiny_model)
    assert infer_vocab_size(state) == tiny_config.model.vocab_size


def test_infer_num_layers(tiny_model, tiny_config):
    state = model_state_dict(tiny_model)
    assert infer_num_layers(state) == tiny_config.model.num_layers


def test_load_training_checkpoint_roundtrip(tmp_path, tiny_model, tiny_config):
    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {
            "epoch": 0,
            "loss": 1.0,
            "model_state": model_state_dict(tiny_model),
            "model_config": tiny_config.model.model_dump(),
        },
        ckpt_path,
    )

    checkpoint = load_training_checkpoint(ckpt_path)
    assert checkpoint.model_config["hidden_size"] == tiny_config.model.hidden_size
    assert checkpoint.epoch == 0


def test_load_model_and_tokenizer_from_checkpoint(tmp_path, tiny_model, tiny_config):
    tokenizer = SimpleCharacterTokenizer(list(string.printable[: tiny_config.model.vocab_size]))
    tokenizer_path = tmp_path / "tokenizer.pt"
    torch.save(tokenizer, tokenizer_path)

    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": model_state_dict(tiny_model),
            "model_config": tiny_config.model.model_dump(),
        },
        ckpt_path,
    )

    config = ServingConfig(
        model_path=str(ckpt_path),
        tokenizer_path=str(tokenizer_path),
        tokenizer_type="simple",
    )
    model, loaded_tokenizer = load_model_and_tokenizer(config)

    assert model.lm_head.out_features == tiny_config.model.vocab_size
    assert loaded_tokenizer.decode(loaded_tokenizer.encode("ab")) == "ab"


def test_load_model_requires_tokenizer_path(tmp_path, tiny_model, tiny_config):
    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {"model_state": model_state_dict(tiny_model), "model_config": tiny_config.model.model_dump()},
        ckpt_path,
    )

    config = ServingConfig(model_path=str(ckpt_path))
    with pytest.raises(ValueError, match="tokenizer_path is required"):
        load_model_and_tokenizer(config)


def test_dummy_fallback_without_checkpoint():
    config = ServingConfig()
    model, tokenizer = load_model_and_tokenizer(config)
    assert model.lm_head.out_features == tokenizer.vocab_size
