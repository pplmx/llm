"""Tests for runtime ModelFactory."""

import pytest
import torch

from llm.models.decoder import DecoderModel
from llm.runtime import MODEL_REGISTRY, ModelFactory


def test_decoder_is_registered():
    assert "decoder" in MODEL_REGISTRY.names()


def test_from_config_builds_decoder(tiny_config):
    model = ModelFactory.from_config(tiny_config.model)
    assert isinstance(model, DecoderModel)
    assert model.lm_head.out_features == tiny_config.model.vocab_size
    assert model.hidden_size == tiny_config.model.hidden_size


def test_build_accepts_overrides(tiny_config):
    model = ModelFactory.from_config(tiny_config.model, max_seq_len=32)
    assert model.max_seq_len == 32


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="not found"):
        ModelFactory.build("unknown_arch")


def test_duplicate_registration_raises():
    with pytest.raises(ValueError, match="already registered"):
        MODEL_REGISTRY.register("decoder", lambda **kwargs: torch.nn.Linear(1, 1))


def test_regression_mlp_is_registered():
    from llm.runtime.bootstrap import ensure_builtins_registered

    ensure_builtins_registered()
    assert "regression_mlp" in MODEL_REGISTRY.names()


def test_from_config_builds_regression_mlp(tiny_config):
    from llm.core.mlp import MLP
    from llm.runtime.bootstrap import ensure_builtins_registered

    ensure_builtins_registered()
    model = ModelFactory.from_config(tiny_config.model, model_type="regression_mlp")
    assert isinstance(model, MLP)
