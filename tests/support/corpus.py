"""Shared text corpora for tokenizer and data tests."""

import string

DEFAULT_INFERENCE_CORPUS = [
    "hello world!",
    "this is a test.",
    "testing cache consistency",
    "你好 世界!",
]

SAMPLE_TEXT_CORPUS = ["abcdefghijklmnopqrstuvwxyz .,<PAD>"]


def printable_corpus(size: int) -> str:
    """Return the first ``size`` printable ASCII characters as a single corpus string."""
    return string.printable[:size]
