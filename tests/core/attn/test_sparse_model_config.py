"""Tests for wiring sparse/streaming attention into the model config (TASK-243).

Proves a ``ModelConfig.attn_sparse`` scheme is a *selectable model option*:
- ``ModelConfig`` validates the kind;
- ``build_config_attention_mask`` turns the config into the repo's SDPA mask-out
  convention (``None`` when unset);
- a ``DecoderModel`` built from such a config via ``ModelFactory.from_config``
  auto-builds the dispatched mask in forward (no manual ``attn_mask`` needed);
- genuinely sparse schemes change decoder output, while a full-coverage scheme
  (sink == every position) is numerically identical to dense attention.
"""

from __future__ import annotations

import pytest
import torch

from llm.core.attn.sparse import build_config_attention_mask
from llm.runtime.model_factory import ModelFactory
from llm.training.core.config import ModelConfig

SEQ, B = 16, 2


def _config(attn_sparse=None, **kwargs) -> ModelConfig:
    base: dict = {
        "vocab_size": 32,
        "hidden_size": 16,
        "num_layers": 2,
        "num_heads": 2,
        "max_seq_len": SEQ,
        "dropout": 0.0,
    }
    if attn_sparse is not None:
        base["attn_sparse"] = attn_sparse
    base.update(kwargs)
    return ModelConfig(**base)


def _decoder(attn_sparse=None) -> torch.nn.Module:
    model = ModelFactory.from_config(_config(attn_sparse=attn_sparse))
    model.eval()  # deterministic (no dropout in eval)
    return model


def test_config_validates_attn_sparse_kind():
    with pytest.raises(ValueError, match="attn_sparse"):
        _config(attn_sparse={"kind": "nope"})
    with pytest.raises(ValueError, match="kind"):
        _config(attn_sparse={"window_size": 4})
    # A valid kind is accepted.
    _config(attn_sparse={"kind": "streaming", "num_sink": 2, "window_size": 4})


def test_helper_returns_none_when_unset():
    assert build_config_attention_mask(_config(), SEQ) is None
    assert build_config_attention_mask(object(), SEQ) is None


def test_helper_returns_mask_out_convention():
    cfg = _config(attn_sparse={"kind": "streaming", "num_sink": 2, "window_size": 4})
    mask = build_config_attention_mask(cfg, SEQ)
    assert mask is not None
    assert mask.shape == (SEQ, SEQ)
    assert mask.dtype == torch.bool
    # True = mask out (SDPA convention): an old non-sink, off-window interior
    # position is blocked here.
    assert bool(mask[10, 5].item())


def test_model_forward_auto_builds_from_config():
    """No manual attn_mask needed — the decoder builds the dispatched mask, and
    ModelFactory.from_config threads ``attn_sparse`` into the model."""
    model = _decoder(attn_sparse={"kind": "streaming", "num_sink": 2, "window_size": 4, "causal": True})
    assert model.attn_sparse == {"kind": "streaming", "num_sink": 2, "window_size": 4, "causal": True}

    inputs = torch.randint(1, 32, (B, SEQ))
    sparse = model(inputs)
    sparse = sparse[0] if isinstance(sparse, tuple) else sparse

    # The same weights without any sparse scheme give a different (dense) result.
    model.attn_sparse = None
    dense = model(inputs)
    dense = dense[0] if isinstance(dense, tuple) else dense
    assert not torch.allclose(sparse, dense, atol=1e-4), "sparse scheme changed nothing in decoder forward"

    # Full-coverage sparse (sink == every position) is identical to dense on the
    # SAME weights — the milestone's CPU parity invariant.
    model.attn_sparse = {"kind": "streaming", "num_sink": SEQ, "window_size": 0, "causal": True}
    full = model(inputs)
    full = full[0] if isinstance(full, tuple) else full
    assert torch.allclose(dense, full, atol=1e-5), "full-coverage sparse must equal dense"
