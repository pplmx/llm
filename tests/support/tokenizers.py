"""Lightweight tokenizer stand-ins for tests that do not need real vocabularies."""


class StubTokenizer:
    """Fixed token-id tokenizer for generation and serving tests."""

    pad_token_id: int = 0
    eos_token_id: int = 99

    def __init__(self, token_ids: list[int] | None = None, decode_char: str = "x"):
        self._token_ids = token_ids or [1, 2, 3]
        self._decode_char = decode_char

    def encode(self, text: str) -> list[int]:
        return list(self._token_ids)

    def decode(self, ids: list[int]) -> str:
        return self._decode_char


class LineTokenizer:
    """Ord-based tokenizer for streaming dataset tests."""

    def __init__(self, modulus: int = 50, pad_token_id: int = 0):
        self.modulus = modulus
        self._pad_token_id = pad_token_id

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    def encode(self, text: str) -> list[int]:
        return [ord(char) % self.modulus for char in text]


class CharBoundTokenizer(LineTokenizer):
    """Ord-based tokenizer with bounded encode length (e.g. PPO prompt tasks)."""

    def __init__(self, max_len: int = 16, modulus: int = 50, eos_id: int = 0, pad_token_id: int = 0):
        super().__init__(modulus=modulus, pad_token_id=pad_token_id)
        self.max_len = max_len
        self.eos_id = eos_id

    def encode(self, text: str) -> list[int]:
        return [ord(c) % self.modulus for c in text[: self.max_len]]

    def decode(self, ids: list[int]) -> str:
        return "x"
