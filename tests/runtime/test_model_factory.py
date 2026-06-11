"""Tests for runtime ModelFactory and bootstrap entry points."""

import pytest
import torch

from llm.runtime.bootstrap import ensure_builtins_registered
from llm.runtime.model_factory import MODEL_REGISTRY, ModelFactory


def test_from_config_builds_decoder(tiny_config):
    ensure_builtins_registered()
    model = ModelFactory.from_config(tiny_config.model)
    assert model.lm_head.out_features == tiny_config.model.vocab_size
    assert model.hidden_size == tiny_config.model.hidden_size
    assert len(model.transformer_blocks) == tiny_config.model.num_layers


def test_build_accepts_overrides(tiny_config):
    ensure_builtins_registered()
    model = ModelFactory.from_config(tiny_config.model, max_seq_len=32)
    assert model.max_seq_len == 32


def test_from_config_builds_regression_mlp(tiny_config):
    ensure_builtins_registered()
    model = ModelFactory.from_config(tiny_config.model, model_type="regression_mlp")
    assert model.hidden_size == tiny_config.model.hidden_size
    assert model.intermediate_size == tiny_config.model.intermediate_size
    assert model(torch.randn(2, tiny_config.model.hidden_size)).shape == (
        2,
        tiny_config.model.hidden_size,
    )


def test_bootstrap_is_idempotent():
    ensure_builtins_registered()
    first = set(MODEL_REGISTRY.names())
    ensure_builtins_registered()
    assert set(MODEL_REGISTRY.names()) == first


def test_unknown_model_type_raises():
    ensure_builtins_registered()
    with pytest.raises(ValueError, match="not found"):
        ModelFactory.build("unknown_arch")


def test_duplicate_registration_raises():
    ensure_builtins_registered()
    with pytest.raises(ValueError, match="already registered"):
        MODEL_REGISTRY.register("decoder", lambda **kwargs: torch.nn.Linear(1, 1))
