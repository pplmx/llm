"""HF tokenizer integration smoke test."""

import pytest

from llm.tokenization.tokenizer import HFTokenizer


def test_hf_tokenizer_round_trips_gpt2():
    """Requirement: HFTokenizer loads gpt2 and round-trips a short phrase."""
    try:
        tokenizer = HFTokenizer.from_pretrained("gpt2")
    except OSError:
        pytest.skip("GPT2 tokenizer download failed (network issue?)")

    text = "Hello world"
    tokens = tokenizer.encode(text)
    assert tokens[:2] == [15496, 995]
    assert tokenizer.decode(tokens) == text
