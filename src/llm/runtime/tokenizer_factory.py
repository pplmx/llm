"""Central tokenizer loading for training, serving, and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, cast

import torch

from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.tokenization.tokenizer import BaseTokenizer, HFTokenizer

DEFAULT_SIMPLE_CORPUS = ["<PAD>", "<EOS>", "<BOS>"]


class TokenizerConfig(Protocol):
    tokenizer_type: str
    tokenizer_path: str | None


class TokenizerFactory:
    """Load tokenizers from training DataConfig or ServingConfig duck-typed objects."""

    @staticmethod
    def from_data_config(
        data_config: TokenizerConfig,
        *,
        default_corpus: list[str] | None = None,
    ) -> BaseTokenizer:
        if data_config.tokenizer_type == "hf":
            if not data_config.tokenizer_path:
                raise ValueError("tokenizer_path must be specified for HF tokenizer.")
            return cast(BaseTokenizer, HFTokenizer.from_pretrained(data_config.tokenizer_path))

        if data_config.tokenizer_path:
            path = Path(data_config.tokenizer_path)
            if not path.exists():
                # A configured-but-missing ``simple`` tokenizer must fail
                # loud, exactly like ``from_serving_config`` (raise
                # FileNotFoundError).  Silently substituting the default
                # corpus tokenizer here means training proceeds with a
                # different vocabulary than the one used to build the data /
                # checkpoint — every epoch trains (and saves) against a
                # tokenizer that can't round-trip the intended vocab, and the
                # mismatch is invisible until serve-time decode errors.
                raise FileNotFoundError(f"Tokenizer file not found: {path}")
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            return cast(BaseTokenizer, loaded)

        corpus = default_corpus or DEFAULT_SIMPLE_CORPUS
        return cast(BaseTokenizer, SimpleCharacterTokenizer(corpus))

    @staticmethod
    def from_serving_config(config: Any) -> Any:
        if config.tokenizer_path:
            path = Path(config.tokenizer_path)
            if config.tokenizer_type == "hf":
                return HFTokenizer.from_pretrained(config.tokenizer_path)
            if not path.exists():
                raise FileNotFoundError(f"Tokenizer file not found: {path}")
            return torch.load(path, map_location="cpu", weights_only=False)

        if getattr(config, "model_path", None):
            raise ValueError("tokenizer_path is required when model_path is set for serving")

        return TokenizerFactory.from_printable_corpus()

    @staticmethod
    def from_printable_corpus() -> SimpleCharacterTokenizer:
        import string

        return SimpleCharacterTokenizer([string.printable])

    @staticmethod
    def from_default_test_corpus() -> SimpleCharacterTokenizer:
        corpus = [
            "hello world",
            "the quick brown fox",
            "testing one two three",
            "abcdefghijklmnopqrstuvwxyz",
        ]
        return SimpleCharacterTokenizer(corpus)

    @staticmethod
    def from_dataset_text(dataset_path: str | Path) -> SimpleCharacterTokenizer:
        """Build a character tokenizer from the unique characters in a text file."""
        text = Path(dataset_path).read_text()
        chars = sorted(set(text))
        corpus = ["<PAD>", "<EOS>", "<BOS>", *chars]
        return SimpleCharacterTokenizer(corpus)

    @staticmethod
    def cache_hf_tokenizer(data_config: TokenizerConfig) -> None:
        if data_config.tokenizer_type == "hf" and data_config.tokenizer_path:
            HFTokenizer.from_pretrained(data_config.tokenizer_path)
