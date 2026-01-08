import pytest
import torch

from llm.inference import generate
from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


@pytest.fixture
def model_and_tokenizer():
    corpus = ["hello world!", "this is a test.", "testing cache consistency"]
    tokenizer = SimpleCharacterTokenizer(corpus)
    # Tiny model
    model = DecoderModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        max_seq_len=64,
    )
    return model, tokenizer


def test_inference_cache_consistency(model_and_tokenizer):
    """Verify that generation with and without cache produces identical results (greedy)."""
    model, tokenizer = model_and_tokenizer
    prompt = "testing"

    # Run with cache (default)
    # Note: generate() currently doesn't expose use_cache, implies True.
    # To test consistency, we need to modify generate() or implement manual loop here.
    # Assuming we will modify generate() to accept use_cache.

    out_cache = generate(model, tokenizer, prompt, max_new_tokens=5, temperature=0.0, use_cache=True)

    out_no_cache = generate(model, tokenizer, prompt, max_new_tokens=5, temperature=0.0, use_cache=False)

    assert out_cache == out_no_cache, f"Cache inconsistency: {out_cache} vs {out_no_cache}"


def test_inference_batch_generation(model_and_tokenizer):
    """Verify that model accepts batch input during manual forward pass."""
    model, tokenizer = model_and_tokenizer
    input_ids = torch.randint(0, tokenizer.vocab_size, (2, 10))

    # Forward pass batch check
    logits = model(input_ids)
    assert logits.shape == (2, 10, tokenizer.vocab_size)

    # Cache batch check
    past_key_values = None
    logits, past_key_values = model(input_ids, use_cache=True)

    # Check return type (logits, kv_caches)
    assert isinstance(past_key_values, list)
    assert len(past_key_values) == len(model.transformer_blocks)

    # Assert model.transformer_blocks access
    assert len(past_key_values) == len(model.transformer_blocks)

    # Update step
    next_token = torch.randint(0, tokenizer.vocab_size, (2, 1))
    logits_next, past_key_values_next = model(next_token, past_key_values=past_key_values, use_cache=True)
    assert logits_next.shape == (2, 1, tokenizer.vocab_size)
