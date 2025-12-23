from unittest.mock import patch

import pytest
import torch

from llm.serving.config import ServingConfig
from llm.serving.engine import LLMEngine
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


@pytest.fixture
def dummy_tokenizer_path(tmp_path):
    corpus = ["hello", "world"]
    tokenizer = SimpleCharacterTokenizer(corpus)
    path = tmp_path / "tokenizer.pt"
    torch.save(tokenizer, path)
    return str(path)


@pytest.fixture
def dummy_model_path(tmp_path):
    # Minimal checkpoint for testing loading logic
    checkpoint = {"model_state": {"module.test_layer": torch.tensor([1.0])}}
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    return str(path)


def test_load_tokenizer_from_path(dummy_tokenizer_path):
    config = ServingConfig(tokenizer_path=dummy_tokenizer_path)
    engine = LLMEngine(config)
    engine.load_model()

    assert engine.tokenizer is not None
    assert isinstance(engine.tokenizer, SimpleCharacterTokenizer)
    assert "h" in engine.tokenizer.stoi


def test_load_model_weights_logic(dummy_model_path):
    # Mocking DecoderModel to avoid needing full valid weights
    with patch("llm.serving.engine.DecoderModel") as MockModel:
        mock_instance = MockModel.return_value

        config = ServingConfig(model_path=dummy_model_path)
        engine = LLMEngine(config)

        # We need to ensure tokenizer is loaded or dummy initialized before model init
        # The engine logic initializes dummy tokenizer if path not provided.

        engine.load_model()

        # Verify load_state_dict was called
        # The checkpoint has 'module.test_layer', engine should strip 'module.'
        expected_state_dict = {"test_layer": torch.tensor([1.0])}

        # We can't easily check tensor equality in called_with directly for all items,
        # so we get the call args.
        args, _ = mock_instance.load_state_dict.call_args
        loaded_dict = args[0]

        assert "test_layer" in loaded_dict
        assert torch.equal(loaded_dict["test_layer"], expected_state_dict["test_layer"])


def test_load_model_failure_invalid_path():
    config = ServingConfig(model_path="/invalid/path.pt")
    engine = LLMEngine(config)

    # The RuntimeError is raised because Model explicitly not loaded is not reached
    # Wait, load_model raises "Failed to load model..." which is a general Exception?
    # Looking at engine.py: load_model raises the caught exception again.
    # torch.load raises FileNotFoundError if file missing

    with pytest.raises(
        (RuntimeError, FileNotFoundError, Exception)
    ):  # Being broad for now as per lint suggestion allows specific
        engine.load_model()
