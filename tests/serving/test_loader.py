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


def test_load_training_checkpoint_v2_split_layout(tmp_path, tiny_model, tiny_config):
    """The serving loader accepts the modern v2 split layout written by
    CheckpointManager — referenced by stem, by sidecar path, or by a
    ``.pt``-suffixed stem (legacy-style path that resolves to the trio)."""
    import logging

    from llm.training.core.checkpoint import CheckpointManager
    from llm.training.core.config import CheckpointConfig

    manager = CheckpointManager(
        CheckpointConfig(checkpoint_dir=str(tmp_path), save_interval=1, keep_last_n=2),
        rank=0,
        logger=logging.getLogger("test-loader"),
    )
    manager.save_checkpoint(
        epoch=0,
        model=tiny_model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        loss=1.5,
        extra_state={"stream_data": {"0": {"line_index": 3}}},
        model_config=tiny_config.model.model_dump(),
    )

    for path in (
        tmp_path / "epoch_1",  # stem (no suffix)
        tmp_path / "epoch_1.safetensors",  # weights sidecar
        tmp_path / "epoch_1.pt",  # legacy-style path, resolves to the trio
    ):
        checkpoint = load_training_checkpoint(path)
        assert checkpoint.model_config["hidden_size"] == tiny_config.model.hidden_size
        assert checkpoint.epoch == 0
        assert checkpoint.loss == 1.5


def test_load_training_checkpoint_missing_raises(tmp_path):
    """A path with neither the legacy nor the v2 layout fails loudly."""
    with pytest.raises(FileNotFoundError):
        load_training_checkpoint(tmp_path / "does-not-exist")


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
