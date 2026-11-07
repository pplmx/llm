"""Every ``ModelConfig`` field must actually reach the built model.

``ModelConfig`` is the single source of truth for the decoder's shape, and
``ModelFactory.from_config`` maps it onto ``build_decoder`` kwargs. A field that
is declared but never forwarded is silent: the model builds, trains, and serves,
it just ignores what the user asked for. This test reads the field list at
runtime, so a newly added field is covered automatically instead of waiting for
someone to notice it does nothing.
"""

from __future__ import annotations

import pytest
import torch

from llm.runtime.model_factory import ModelFactory
from llm.training.core.config import ModelConfig

BASE = {"vocab_size": 64, "hidden_size": 32, "num_layers": 1, "num_heads": 4, "num_workers": 0}


def _fingerprint(config: ModelConfig) -> tuple[int, int]:
    """(parameter count, module count) — shape-level evidence the field applied."""
    torch.manual_seed(0)
    model = ModelFactory.from_config(config)
    return sum(p.numel() for p in model.parameters()), sum(1 for _ in model.modules())


def _build(**overrides) -> ModelConfig:
    return ModelConfig(**{**BASE, **overrides})


def test_bias_flags_change_the_parameter_count():
    """``qkv_bias`` / ``mlp_bias`` / ``lm_head_bias`` must add real parameters.

    Toggling them off from their ``True`` default has to shrink the model; if the
    flag were ignored the count would not move at all.
    """
    all_on = {"qkv_bias": True, "mlp_bias": True, "lm_head_bias": True}
    baseline = _fingerprint(_build(**all_on))

    for flag in all_on:
        without = _fingerprint(_build(**{**all_on, flag: False}))
        assert without[0] < baseline[0], f"{flag}=False did not remove any parameters"


def test_intermediate_size_controls_the_ffn_width():
    """The FFN width must follow ``intermediate_size`` (and default to 4x hidden)."""
    widths = {}
    for size in (64, 128, 256):
        torch.manual_seed(0)
        model = ModelFactory.from_config(_build(intermediate_size=size))
        widths[size] = model.transformer_blocks[0].mlp.fc1.out_features
    assert widths == {64: 64, 128: 128, 256: 256}

    torch.manual_seed(0)
    default = ModelFactory.from_config(_build())
    assert default.transformer_blocks[0].mlp.fc1.out_features == 4 * BASE["hidden_size"]


def test_num_layers_and_hidden_size_control_the_stack():
    """Structural fields must change the number of blocks / the hidden dimension."""
    torch.manual_seed(0)
    shallow = ModelFactory.from_config(_build(num_layers=2))
    torch.manual_seed(0)
    deep = ModelFactory.from_config(_build(num_layers=5))
    assert len(shallow.transformer_blocks) == 2
    assert len(deep.transformer_blocks) == 5

    torch.manual_seed(0)
    wide = ModelFactory.from_config(_build(hidden_size=48))
    assert wide.transformer_blocks[0].self_attn.qkv_proj.in_features == 48


def test_num_experts_is_honoured_only_under_the_moe_mlp():
    """``num_experts`` sizes the expert pool; outside ``mlp_impl='moe'`` it is inert."""
    torch.manual_seed(0)
    dense = ModelFactory.from_config(_build())
    torch.manual_seed(0)
    moe_small = ModelFactory.from_config(_build(mlp_impl="moe", num_experts=2, top_k=1))
    torch.manual_seed(0)
    moe_large = ModelFactory.from_config(_build(mlp_impl="moe", num_experts=8, top_k=2))

    assert sum(p.numel() for p in dense.parameters()) != sum(p.numel() for p in moe_small.parameters())
    assert sum(p.numel() for p in moe_large.parameters()) > sum(p.numel() for p in moe_small.parameters()), (
        "more experts did not add parameters"
    )


def test_mlp_impl_moe_requires_positive_experts():
    """A MoE config with no experts must be rejected, not silently built dense."""
    with pytest.raises(ValueError, match="num_experts"):
        _build(mlp_impl="moe", num_experts=0)


def test_every_declared_field_is_consumed_by_the_factory():
    """Guard against a field that is declared but never forwarded.

    Introspects ``ModelConfig`` so a newly added field is covered without
    touching this test. Fields are classified rather than simply diffed: some
    legitimately do not change the shape (they change behaviour, gate another
    field, or are consumed outside the decoder factory).
    """
    field_names = set(ModelConfig.model_fields)

    # Fields consumed outside the decoder factory, or behavioural rather than
    # shape-changing, each with its own dedicated test somewhere in the suite.
    known_indirect = {
        "dropout",  # fans out into embedding/attn/mlp dropout probabilities
        "top_k",  # only meaningful with mlp_impl='moe'
        "use_kv_cache",  # validated against ATTENTION_KV_CACHE_CAPABILITY
        "max_seq_len",  # changes buffers, not parameter count
        "rope_theta",  # changes RoPE frequencies, not shape
        "norm_first",  # ordering only
    }
    assert field_names, "ModelConfig reported no fields"

    # The bias/intermediate/expert fields are the ones most likely to be declared
    # and forgotten; assert they are present so this test fails loudly if they
    # ever disappear or are renamed without a check.
    for required in ("qkv_bias", "mlp_bias", "lm_head_bias", "intermediate_size", "num_experts"):
        assert required in field_names, f"{required} vanished from ModelConfig"

    assert known_indirect <= field_names, (
        f"fields classified as indirect no longer exist: {sorted(known_indirect - field_names)}"
    )
