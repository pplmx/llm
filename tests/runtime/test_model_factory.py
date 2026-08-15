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


def test_from_config_roundtrips_model_defining_flags(tiny_config):
    """RIL ISS-129: model-defining flags (RoPE + bias-free, as real
    Llama/Mistral use) must survive the ModelConfig -> kwargs -> model trip.

    Before this fix, ``decoder_kwargs_from_config`` only mapped the original
    ModelConfig fields, so a model built with ``use_rope=True,
    qkv_bias=False, mlp_bias=False, lm_head_bias=False`` was silently
    REBUILT with the defaults (no RoPE, with biases) when loaded from a
    checkpoint's ``model_config`` sidecar at serving time — every
    *_proj.bias / lm_head.bias sat at random init and the position encoding
    was swapped. The flags now live on ModelConfig, persist into the
    checkpoint sidecar, and are mapped back here.
    """
    ensure_builtins_registered()
    cfg = tiny_config.model.model_copy(deep=True)
    cfg.use_rope = True
    cfg.rope_theta = 47.0
    cfg.qkv_bias = False
    cfg.mlp_bias = False
    cfg.lm_head_bias = False
    cfg.norm_first = False
    cfg.pos_encoding_learned = False

    model = ModelFactory.from_config(cfg)
    assert model.use_rope is True, "use_rope must be carried into the model"
    assert model.rope_theta == 47.0
    assert model.qkv_bias is False
    assert model.mlp_bias is False
    assert model.lm_head_bias is False
    assert model.norm_first is False

    # And the reverse path used at RESUME-time config comparison: the fields
    # survive model_dump so the checkpoint sidecar carries them.
    dumped = cfg.model_dump()
    assert dumped["use_rope"] is True
    assert dumped["qkv_bias"] is False


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
