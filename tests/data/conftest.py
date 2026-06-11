"""Shared fixtures for data-layer tests."""

import pytest

from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from tests.support.corpus import SAMPLE_TEXT_CORPUS
from tests.support.tokenizers import LineTokenizer


@pytest.fixture(scope="session")
def sample_text_tokenizer():
    """Session-scoped character tokenizer with a fixed printable corpus."""
    return SimpleCharacterTokenizer(SAMPLE_TEXT_CORPUS)


@pytest.fixture
def line_tokenizer():
    return LineTokenizer()
