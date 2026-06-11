"""Tests for TokenizerFactory."""

import pytest
import torch

from llm.runtime.tokenizer_factory import TokenizerFactory
from llm.serving.config import ServingConfig
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.training.core.config import Config


def test_from_data_config_simple_fallback():
    config = Config()
    tokenizer = TokenizerFactory.from_data_config(config.data)
    assert tokenizer.vocab_size > 3
    assert tokenizer.decode(tokenizer.encode("<PAD>")) == "<PAD>"


def test_from_data_config_pickle(tmp_path):
    config = Config()
    tokenizer = SimpleCharacterTokenizer(["abc"])
    path = tmp_path / "tok.pt"
    torch.save(tokenizer, path)
    config.data.tokenizer_path = str(path)

    loaded = TokenizerFactory.from_data_config(config.data)
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.encode("abc") == tokenizer.encode("abc")


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
    roundtrip = tokenizer.decode(tokenizer.encode("Hello"))
    assert roundtrip == "Hello"


def test_from_dataset_text_round_trips_file_contents(tmp_path):
    """Requirement: from_dataset_text builds a vocab covering every character in the file."""
    data_file = tmp_path / "corpus.txt"
    data_file.write_text("cab", encoding="utf-8")

    tokenizer = TokenizerFactory.from_dataset_text(data_file)

    assert tokenizer.decode(tokenizer.encode("cab")) == "cab"
    assert len(tokenizer.encode("cab")) == 3
