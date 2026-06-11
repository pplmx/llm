"""Tests for TokenizerFactory."""


import pytest
import torch

from llm.runtime.tokenizer_factory import DEFAULT_SIMPLE_CORPUS, TokenizerFactory
from llm.serving.config import ServingConfig
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.training.core.config import Config


def test_from_data_config_simple_fallback():
    config = Config()
    tokenizer = TokenizerFactory.from_data_config(config.data)
    assert isinstance(tokenizer, SimpleCharacterTokenizer)
    assert tokenizer.encode("<PAD>")


def test_from_data_config_pickle(tmp_path):
    config = Config()
    tokenizer = SimpleCharacterTokenizer(["abc"])
    path = tmp_path / "tok.pt"
    torch.save(tokenizer, path)
    config.data.tokenizer_path = str(path)

    loaded = TokenizerFactory.from_data_config(config.data)
    assert loaded.vocab_size == tokenizer.vocab_size


def test_from_data_config_hf_requires_path():
    config = Config()
    config.data.tokenizer_type = "hf"
    config.data.tokenizer_path = None

    with pytest.raises(ValueError, match="tokenizer_path"):
        TokenizerFactory.from_data_config(config.data)


def test_from_serving_config_requires_tokenizer_with_model(tmp_path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_text("x", encoding="utf-8")
    config = ServingConfig(model_path=str(ckpt))

    with pytest.raises(ValueError, match="tokenizer_path is required"):
        TokenizerFactory.from_serving_config(config)


def test_from_serving_config_default_simple():
    config = ServingConfig()
    tokenizer = TokenizerFactory.from_serving_config(config)
    assert isinstance(tokenizer, SimpleCharacterTokenizer)
