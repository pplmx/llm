"""``FunctionalKVCache`` must match ``KVCache`` bit-for-bit.

ADR-016 measured why the export path forces ``use_cache=False``: the production
cache writes into a pre-allocated buffer (a side effect), and a side effect
cannot be recorded in an exported graph, so a decoded token recomputes the whole
prefix (248 positions versus 24, a 10.3x gap at prompt=8/new=16).

``FunctionalKVCache`` keeps the same interface but builds the result with
``torch.cat``, making the update a pure function. That is a prerequisite for
exporting a cache as explicit graph I/O — but only useful if it computes exactly
the same thing, so equivalence is pinned here with the same bit-exact standard
the other ZeRO/cache tests use.

What is NOT yet demonstrated (and is recorded in the ADR rather than pretended
away): a full incremental-decode export. Tracing the explicit-cache-I/O entry
point succeeds and the prefill matches eager to 4.8e-7, but decode steps diverge
(~0.34) because ``has_past`` is decided by a Python bool that the tracer freezes
into a constant, so the graph keeps using prefill's causal setting. Passing an
explicit additive mask instead is the obvious fix and it hit an SDPA broadcast
shape error. That work is unfinished; these tests cover the part that IS correct.
"""

from __future__ import annotations

import pytest
import torch

from llm.core.kv_cache import FunctionalKVCache, KVCache
from llm.models.decoder import DecoderModel

VOCAB = 64
HIDDEN = 32
LAYERS = 2
HEADS = 2
MAX_SEQ = 32


def _model() -> DecoderModel:
    torch.manual_seed(0)
    return DecoderModel(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_layers=LAYERS, num_heads=HEADS, max_seq_len=MAX_SEQ
    ).eval()


def _cache_shape(model: DecoderModel) -> tuple[int, int]:
    """Resolve ``(num_kv_heads, head_dim)`` from the real module, not a guess."""
    block = model.transformer_blocks[0]
    attn = getattr(block, "attention", None) or getattr(block, "self_attn", None)
    assert attn is not None, "the decoder must expose its attention module for this test to size the cache"
    num_kv = getattr(attn, "num_kv_heads", HEADS)
    head_dim = model.embedding_layer.token_embeddings.embedding_dim // HEADS
    return num_kv, head_dim


def _run(model: DecoderModel, cache_cls, prompt: torch.Tensor, steps: list[torch.Tensor]):
    num_kv, head_dim = _cache_shape(model)
    caches = [cache_cls(1, MAX_SEQ, num_kv, head_dim) for _ in range(LAYERS)]
    outputs = []
    with torch.no_grad():
        out = model(prompt, use_cache=True, kv_caches=caches)
        outputs.append(out[0] if isinstance(out, tuple) else out)
        for token in steps:
            out = model(token, use_cache=True, kv_caches=caches)
            outputs.append(out[0] if isinstance(out, tuple) else out)
    return outputs, caches


@pytest.mark.quick
def test_functional_cache_matches_the_in_place_cache_bit_exactly():
    """Prefill + N decode steps must agree to the last bit, not approximately.

    The functional cache is a re-expression, not an approximation: it returns the
    same keys/values in the same order, so the model's outputs must be identical.
    """
    model = _model()
    torch.manual_seed(5)
    prompt = torch.randint(0, VOCAB, (1, 4))
    steps = [torch.randint(0, VOCAB, (1, 1)) for _ in range(3)]

    in_place, in_place_caches = _run(model, KVCache, prompt, steps)
    functional, functional_caches = _run(model, FunctionalKVCache, prompt, steps)

    assert len(in_place) == len(functional) == len(steps) + 1
    for step, (want, got) in enumerate(zip(in_place, functional, strict=True)):
        assert torch.equal(got, want), (
            f"step {step}: functional cache diverged from the in-place cache "
            f"(max |d|={(got - want).abs().max().item():.3e})"
        )

    assert [c.seq_len for c in functional_caches] == [c.seq_len for c in in_place_caches], (
        "the two caches must agree on how much history they hold"
    )


@pytest.mark.quick
def test_functional_cache_returns_exactly_the_tokens_seen():
    """No pre-allocated capacity: the returned K/V length equals the tokens fed.

    This is what lets an exporter treat the cache length as a function of the
    inputs. The in-place cache returns a view over ``max_seq_len`` capacity and
    slices it, which a tracer cannot express as a dynamic shape.
    """
    num_kv, head_dim = 2, HIDDEN // HEADS
    cache = FunctionalKVCache(1, MAX_SEQ, num_kv, head_dim)

    for step in range(1, 4):
        k = torch.randn(1, num_kv, 1, head_dim)
        v = torch.randn(1, num_kv, 1, head_dim)
        k_all, v_all = cache.update(k, v)
        assert k_all.shape == (1, num_kv, step, head_dim), (
            f"after {step} token(s) the cache must hold exactly {step} positions, got {tuple(k_all.shape)}"
        )
        assert v_all.shape == k_all.shape
        assert cache.seq_len == step


@pytest.mark.quick
def test_functional_cache_refuses_overflow_like_the_in_place_cache():
    """Same guard, same message shape: both caches must reject more than max_seq_len."""
    num_kv, head_dim = 2, HIDDEN // HEADS
    cache = FunctionalKVCache(1, 4, num_kv, head_dim)
    good = torch.randn(1, num_kv, 4, head_dim)
    cache.update(good, good)

    extra = torch.randn(1, num_kv, 1, head_dim)
    with pytest.raises(ValueError, match="Cache overflow"):
        cache.update(extra, extra)


@pytest.mark.quick
def test_functional_cache_reset_clears_history():
    """``reset`` must drop the history so a reused cache starts from zero."""
    num_kv, head_dim = 2, HIDDEN // HEADS
    cache = FunctionalKVCache(1, MAX_SEQ, num_kv, head_dim)
    chunk = torch.randn(1, num_kv, 2, head_dim)
    cache.update(chunk, chunk)
    assert cache.seq_len == 2

    cache.reset()
    assert cache.seq_len == 0
    k_all, _ = cache.update(chunk[:, :, :1], chunk[:, :, :1])
    assert k_all.shape[2] == 1, f"after reset the history must restart, got {k_all.shape[2]} positions"
