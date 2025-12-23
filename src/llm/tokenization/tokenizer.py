from typing import Protocol


class BaseTokenizer(Protocol):
    """
    Abstract base class for all tokenizers.
    """

    vocab_size: int
    pad_token_id: int | None
    bos_token_id: int | None
    eos_token_id: int | None

    def encode(self, text: str) -> list[int]:
        """Encodes a string into a list of token IDs."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decodes a list of token IDs back into a string."""
        ...


class HFTokenizer:
    """
    Wrapper for HuggingFace Transformers Tokenizers.
    """

    def __init__(self, model_name_or_path: str):
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Ensure pad token exists
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int | None:
        return self._tokenizer.pad_token_id

    @property
    def bos_token_id(self) -> int | None:
        return self._tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        return self._tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        # Return simple list of ints
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)

    def save_pretrained(self, save_directory: str):
        self._tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, path: str) -> "HFTokenizer":
        return cls(path)
