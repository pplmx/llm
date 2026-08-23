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
from llm.models.decoder import DecoderModel
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


def test_sparse_scheme_rejects_flash_attn_backend():
    """RIL TASK-248: the flash_attn backend ignores ``attn_mask``, so a sparse
    scheme with it would silently run dense attention. The model refuses loudly
    instead of building a scheme that has no effect. (mha/mla route through the
    SDPA wrapper and do consume the mask, so they remain supported.)"""
    # mha + sparse stays supported (goes through sdpa).
    _decoder(attn_sparse={"kind": "streaming", "num_sink": 2, "window_size": 4})
    with pytest.raises(NotImplementedError, match="flash_attn"):
        DecoderModel(
            vocab_size=32,
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            max_seq_len=SEQ,
            attn_impl="flash_attn",
            attn_sparse={"kind": "streaming", "num_sink": 2, "window_size": 4},
            embedding_dropout_p=0.0,
            attn_dropout_p=0.0,
            mlp_dropout_p=0.0,
        )


def test_helper_returns_mask_out_convention():
    cfg = _config(attn_sparse={"kind": "streaming", "num_sink": 2, "window_size": 4})
    mask = build_config_attention_mask(cfg, SEQ)
    assert mask is not None
    assert mask.shape == (SEQ, SEQ)
    assert mask.dtype == torch.bool
    # True = mask out (SDPA convention): an old non-sink, off-window interior
    # position is blocked here.
    assert bool(mask[10, 5].item())


def test_helper_builds_rectangular_key_history_mask():
    """RIL TASK-245: at a KV-cache decode step the mask is [Sq, Sk] over the key
    history (Sk >= Sq), so sink/window can constrain the accumulated past keys
    instead of a degenerate square of the 1-token input."""
    cfg = _config(attn_sparse={"kind": "streaming", "num_sink": 2, "window_size": 4})
    mask = build_config_attention_mask(cfg, 1, key_len=17)
    assert mask.shape == (1, 17)
    assert mask.dtype == torch.bool
    # The single query sits at absolute position 16: sink keys {0,1} are kept,
    # and an old non-sink, off-window key (e.g. position 5) is masked out.
    assert not bool(mask[0, 0].item())  # sink kept
    assert bool(mask[0, 1].item()) is False  # second sink kept
    assert bool(mask[0, 5].item())  # old non-sink, off-window -> masked out


def test_decoder_decode_uses_key_history_mask():
    """RIL TASK-245: during KV-cache decode the sparse scheme constrains the
    accumulated keys (sparse decode differs from dense), and a full-coverage
    scheme keeps the decode dense-equivalent. KV is held fixed via a dense
    prefill; only the decode-time scheme varies."""
    from llm.core.kv_cache import create_decoder_kv_caches

    scheme = {"kind": "streaming", "num_sink": 2, "window_size": 4, "causal": True}
    # Larger context so decoding one token past the 16-pre/mid prompt stays in
    # range (the end-of-sequence check is max_seq_len based).
    model = ModelFactory.from_config(_config(attn_sparse=None, max_seq_len=64))
    model.eval()
    pre = torch.randint(1, 32, (1, SEQ))
    nxt = torch.randint(1, 32, (1, 1))

    def _decode(attn_sparse):
        model.attn_sparse = None  # dense prefill -> identical KV each run
        kvs = create_decoder_kv_caches(model, batch_size=1)
        _, kvs = model(pre, kv_caches=kvs, use_cache=True)
        model.attn_sparse = attn_sparse  # scheme drives only the decode step
        out, _ = model(nxt, kv_caches=kvs, use_cache=True)
        return out[0] if isinstance(out, tuple) else out

    dense = _decode(None)
    sparse = _decode(scheme)
    assert not torch.allclose(sparse, dense, atol=1e-6), "sparse scheme changed nothing in decode"

    full = _decode({"kind": "streaming", "num_sink": 64, "window_size": 0, "causal": True})
    assert torch.allclose(dense, full, atol=1e-6), "full-coverage sparse decode must equal dense decode"


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
