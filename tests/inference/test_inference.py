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
    input_ids = torch.randint(0, tokenizer.vocab_size, (2, 10))

    logits = model(input_ids)
    assert logits.shape == (2, 10, tokenizer.vocab_size)

    kv_caches = create_decoder_kv_caches(model, batch_size=2)
    logits, kv_caches = model(input_ids, kv_caches=kv_caches, use_cache=True)
    assert len(kv_caches) == len(model.transformer_blocks)

    next_token = torch.randint(0, tokenizer.vocab_size, (2, 1))
    logits_next, kv_caches = model(next_token, kv_caches=kv_caches, use_cache=True)
    assert logits_next.shape == (2, 1, tokenizer.vocab_size)
