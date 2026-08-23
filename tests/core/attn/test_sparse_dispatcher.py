"""Tests for the unified sparse-attention dispatcher (TASK-242)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

from llm.core.attn.big_bird import build_bigbird_mask
from llm.core.attn.block_sparse import build_block_sparse_mask, mask_to_additive
from llm.core.attn.longformer import build_longformer_mask
from llm.core.attn.sdpa import sdpa
from llm.core.attn.sparse import SUPPORTED_KINDS, build_sparse_attention_mask
from llm.core.attn.streaming_llm import build_streamingllm_mask

SEQ, B, N, D = 24, 2, 4, 16

_KINDS: dict[str, dict] = {
    "block_sparse": {"block_size": 4, "window_blocks": 1, "global_blocks": 2, "random_blocks": 1, "seed": 1},
    "streaming": {"num_sink": 2, "window_size": 6},
    "longformer": {"window_size": 6, "dilation": 2, "num_global": 2},
    "bigbird": {
        "block_size": 4,
        "num_global_blocks": 2,
        "window_blocks": 1,
        "num_random_blocks": 1,
        "seed": 1,
    },
}
_DIRECT = {
    "block_sparse": build_block_sparse_mask,
    "streaming": build_streamingllm_mask,
    "longformer": build_longformer_mask,
    "bigbird": build_bigbird_mask,
}
_INVALID = {
    "block_sparse": {"block_size": 0},  # must be >= 1
    "streaming": {"window_size": -1},
    "longformer": {"dilation": 0},
    "bigbird": {"block_size": 0},
}


def test_supported_kinds_registry():
    assert set(SUPPORTED_KINDS) == set(_KINDS)


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown sparse attention kind"):
        build_sparse_attention_mask("does_not_exist", SEQ)
    with pytest.raises(ValueError, match="block_sparse"):
        build_sparse_attention_mask("nope", SEQ)


@pytest.mark.parametrize("kind", list(_KINDS))
def test_dispatcher_delegates_equals_direct(kind: str):
    dispatched = build_sparse_attention_mask(kind, SEQ, **_KINDS[kind])
    direct = _DIRECT[kind](SEQ, **_KINDS[kind])
    assert torch.equal(dispatched, direct)


@pytest.mark.parametrize("kind", list(_KINDS))
def test_dispatcher_propagates_builder_validation(kind: str):
    bad = dict(_KINDS[kind])
    bad.update(_INVALID[kind])
    with pytest.raises(ValueError, match="must be"):
        build_sparse_attention_mask(kind, SEQ, **bad)


@pytest.mark.parametrize("kind", list(_KINDS))
def test_every_kind_through_real_attention_matches_explicit_masking(kind: str):
    torch.manual_seed(0)
    q = torch.randn(B, N, SEQ, D)
    k = torch.randn(B, N, SEQ, D)
    v = torch.randn(B, N, SEQ, D)
    mask = build_sparse_attention_mask(kind, SEQ, **_KINDS[kind])
    out = sdpa(q, k, v, attn_mask=(~mask).bool().unsqueeze(0).unsqueeze(0), is_causal=False)
    additive = mask_to_additive(mask).to(q.dtype).unsqueeze(0).unsqueeze(0)
    ref = functional.scaled_dot_product_attention(q, k, v, attn_mask=additive, is_causal=False)
    assert torch.allclose(out, ref, atol=1e-5)
