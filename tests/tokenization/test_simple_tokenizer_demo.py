"""
Simple Character Tokenizer Demo Tests

Tests SimpleCharacterTokenizer encoding, decoding, and error handling.
"""

import pytest

from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


def test_tokenizer_basic():
    """Test basic encoding and decoding."""
    corpus = ["hello world!", "你好 世界!"]
    tokenizer = SimpleCharacterTokenizer(corpus)

    # Check vocabulary
    assert tokenizer.vocab_size > 0
    assert len(tokenizer.chars) == tokenizer.vocab_size
    assert len(tokenizer.stoi) == tokenizer.vocab_size
    assert len(tokenizer.itos) == tokenizer.vocab_size

    # Check PAD token
    assert tokenizer.itos[tokenizer.pad_token_id] == tokenizer.pad_char
    assert tokenizer.pad_char in tokenizer.stoi
    assert tokenizer.pad_char in tokenizer.chars


def test_tokenizer_encode_decode():
    """Test encoding and decoding round-trip."""
    corpus = ["hello world!", "你好 世界!"]
    tokenizer = SimpleCharacterTokenizer(corpus)

    # Test text 1
    text1 = "hello!"
    encoded1 = tokenizer.encode(text1)
    decoded1 = tokenizer.decode(encoded1)
    assert decoded1 == text1

    # Test text 2
    text2 = "你好!"
    encoded2 = tokenizer.encode(text2)
    decoded2 = tokenizer.decode(encoded2)
    assert decoded2 == text2


def test_tokenizer_empty_string():
    """Test encoding empty string and decoding empty list."""
    corpus = ["hello world!", "你好 世界!"]
    tokenizer = SimpleCharacterTokenizer(corpus)

    # Empty string encoding
    empty_encoded = tokenizer.encode("")
    assert empty_encoded == []

    # Empty list decoding
    empty_decoded = tokenizer.decode([])
    assert empty_decoded == ""


def test_tokenizer_unknown_character():
    """Test that unknown characters raise KeyError."""
    corpus = ["hello world!", "你好 世界!"]
    tokenizer = SimpleCharacterTokenizer(corpus)

    # 'a', 'b', 'c' are not in the example corpus
    with pytest.raises(KeyError):
        tokenizer.encode("abc")


def test_tokenizer_unknown_token_id():
    """Test that unknown token IDs raise KeyError."""
    corpus = ["hello world!", "你好 世界!"]
    tokenizer = SimpleCharacterTokenizer(corpus)

    # An ID guaranteed to be out of vocab
    unknown_id = tokenizer.vocab_size + 10
    with pytest.raises(KeyError):
        tokenizer.decode([unknown_id])


def test_tokenizer_pad_token():
    """Test decoding PAD token."""
    corpus = ["hello world!", "你好 世界!"]
    tokenizer = SimpleCharacterTokenizer(corpus)

    decoded_pad = tokenizer.decode([tokenizer.pad_token_id])
    assert decoded_pad == tokenizer.pad_char
