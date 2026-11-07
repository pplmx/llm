"""Tests for generation entry point."""

import pytest
import torch

from llm.core.kv_cache import create_decoder_kv_caches
from llm.generation import generate


@pytest.mark.slow
def test_generate_greedy_appends_exact_token_count(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "hello"
    max_new_tokens = 5

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0,
    )

    encoded_prompt = tokenizer.encode(prompt)
    encoded_output = tokenizer.encode(output)
    assert len(encoded_output) == len(encoded_prompt) + max_new_tokens


@pytest.mark.slow
def test_generate_sampling_appends_exact_token_count(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "test"
    max_new_tokens = 3

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=2,
    )

    encoded_prompt = tokenizer.encode(prompt)
    encoded_output = tokenizer.encode(output)
    assert len(encoded_output) == len(encoded_prompt) + max_new_tokens


@pytest.mark.slow
def test_generate_preserves_truncated_prompt_length_in_output(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "a" * 100
    max_new_tokens = 2

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0,
    )

    encoded_output = tokenizer.encode(output)
    assert len(encoded_output) == 100 + max_new_tokens


def test_generate_cache_matches_no_cache(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "testing"

    out_cache = generate(model, tokenizer, prompt, max_new_tokens=5, temperature=0.0, use_cache=True)
    out_no_cache = generate(model, tokenizer, prompt, max_new_tokens=5, temperature=0.0, use_cache=False)

    assert out_cache == out_no_cache


def test_model_batch_forward_and_kv_cache(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    device = next(model.parameters()).device
    input_ids = torch.randint(0, tokenizer.vocab_size, (2, 10), device=device)

    logits = model(input_ids)
    assert logits.shape == (2, 10, tokenizer.vocab_size)

    kv_caches = create_decoder_kv_caches(model, batch_size=2)
    logits, kv_caches = model(input_ids, kv_caches=kv_caches, use_cache=True)
    assert len(kv_caches) == len(model.transformer_blocks)

    next_token = torch.randint(0, tokenizer.vocab_size, (2, 1), device=device)
    logits_next, kv_caches = model(next_token, kv_caches=kv_caches, use_cache=True)
    assert logits_next.shape == (2, 1, tokenizer.vocab_size)


# --- KV-cache numerical equivalence (shape checks are not enough) ---


def _eval_model(model_and_tokenizer):
    """The fixture model is in TRAIN mode, so dropout makes forwards differ.

    Every test below compares two forward passes; without `eval()` the dropout
    mask is redrawn each time and the comparison fails on randomness rather
    than on the property under test (verified: two forwards of the fixture model
    are NOT identical in train mode, identical in eval mode).
    """
    model, tokenizer = model_and_tokenizer
    model.eval()
    return model, tokenizer


def test_kv_cache_prefill_matches_a_full_forward(model_and_tokenizer):
    """Prefilling through a cache must produce the SAME logits as a plain forward.

    ``test_model_batch_forward_and_kv_cache`` feeds a prompt through
    ``kv_caches=...`` but only asserts shapes, so a cache path that computed
    different values would still pass. Numerics is the whole point of the cache:
    every generated token comes from these logits.
    """
    model, tokenizer = _eval_model(model_and_tokenizer)
    device = next(model.parameters()).device
    input_ids = torch.randint(0, tokenizer.vocab_size, (2, 10), device=device)

    with torch.no_grad():
        plain = model(input_ids)
        kv_caches = create_decoder_kv_caches(model, batch_size=2)
        cached, _ = model(input_ids, kv_caches=kv_caches, use_cache=True)

    assert torch.allclose(plain, cached, atol=1e-5), (
        f"cached prefill diverged from a plain forward: max|delta|={(plain - cached).abs().max().item():.2e}"
    )


def test_kv_cache_incremental_decode_matches_a_full_forward(model_and_tokenizer):
    """Token-by-token decoding through the cache must match one full forward.

    The strongest form of the property: feed the prompt one token at a time with
    explicit ``position_ids`` and compare the FINAL position's logits against a
    single full-sequence forward. A stale or mis-indexed cache shows up here as
    a large delta, whereas the shape assertions elsewhere cannot see it.
    """
    model, tokenizer = _eval_model(model_and_tokenizer)
    device = next(model.parameters()).device
    seq_len = 8
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device=device)

    with torch.no_grad():
        full = model(input_ids)

    kv_caches = create_decoder_kv_caches(model, batch_size=1)
    last = None
    for position in range(seq_len):
        with torch.no_grad():
            last, kv_caches = model(
                input_ids[:, position : position + 1],
                kv_caches=kv_caches,
                use_cache=True,
                position_ids=torch.tensor([[position]], device=device),
            )

    assert last is not None
    assert torch.allclose(last[:, -1], full[:, -1], atol=1e-5), (
        "incremental decode diverged from a full forward: "
        f"max|delta|={(last[:, -1] - full[:, -1]).abs().max().item():.2e}"
    )


def test_each_position_of_a_full_forward_matches_its_prefix(model_and_tokenizer):
    """Causality invariant: position t must not depend on tokens after t.

    ``logits(ids[:, :t])[:, -1]`` must equal ``logits(ids)[:, t-1]``. A mask bug
    that lets a position see the future breaks this immediately, and it is
    independent of the cache implementation.
    """
    model, tokenizer = _eval_model(model_and_tokenizer)
    device = next(model.parameters()).device
    seq_len = 8
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device=device)

    with torch.no_grad():
        full = model(input_ids)
        for position in range(1, seq_len + 1):
            prefix = model(input_ids[:, :position])
            assert torch.allclose(prefix[:, -1], full[:, position - 1], atol=1e-5), (
                f"position {position - 1} changed when tokens after it were appended — "
                "the causal mask is leaking the future"
            )
