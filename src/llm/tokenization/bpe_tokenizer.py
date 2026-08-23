from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

DEFAULT_UNK_TOKEN = "".join(("<", "unk", ">"))


class BPETokenizer:
    """
    A Byte Pair Encoding (BPE) tokenizer using the `tokenizers` library.
    """

    def __init__(self, tokenizer: Tokenizer | None = None):
        """
        Initializes the BPETokenizer.

        Args:
            tokenizer (Tokenizer, optional): An existing tokenizers.Tokenizer instance.
        """
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = Tokenizer(models.BPE(unk_token=DEFAULT_UNK_TOKEN))
            self.tokenizer.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Lowercase()])
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = decoders.ByteLevel()

    @classmethod
    def train(
        cls,
        files: list[str],
        vocab_size: int = 5000,
        min_frequency: int = 2,
        special_tokens: list[str] | None = None,
    ) -> BPETokenizer:
        """
        Trains a BPE tokenizer on the given files.

        Args:
            files (list[str]): List of paths to text files for training.
            vocab_size (int): The desired vocabulary size.
            min_frequency (int): The minimum frequency for a pair to be merged.
            special_tokens (list[str]): List of special tokens to include.

        Returns:
            BPETokenizer: A trained tokenizer instance.
        """
        if special_tokens is None:
            special_tokens = [DEFAULT_UNK_TOKEN, "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

        tokenizer = Tokenizer(models.BPE(unk_token=DEFAULT_UNK_TOKEN))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Lowercase()])
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        tokenizer.train(files, trainer)
        return cls(tokenizer)

    def encode(self, text: str) -> list[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            text (str): The input text.

        Returns:
            list[int]: The list of token IDs.
        """
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of token IDs back into a string.

        Args:
            ids (list[int]): The list of token IDs.
            skip_special_tokens (bool): Whether to skip special tokens in the output.

        Returns:
            str: The decoded string.
        """
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: str) -> None:
        """
        Saves the tokenizer to a file.

        Args:
            path (str): The path to save the tokenizer to.
        """
        p = Path(path)
        if not p.parent.exists():
            p.parent.mkdir(parents=True)
        self.tokenizer.save(str(p))

    @classmethod
    def load(cls, path: str) -> BPETokenizer:
        """
        Loads a tokenizer from a file.

        Args:
            path (str): The path to the tokenizer file.

        Returns:
            BPETokenizer: The loaded tokenizer instance.
        """
        return cls(Tokenizer.from_file(path))

    @property
    def vocab_size(self) -> int:
        """
        Returns the vocabulary size.
        """
        return self.tokenizer.get_vocab_size()

    def get_vocab(self) -> dict:
        """
        Returns the vocabulary mapping.
        """
        return self.tokenizer.get_vocab()

    @property
    def pad_token_id(self) -> int:
        """
        Returns the ID of the [PAD] token.

        ``token_to_id("[PAD]")`` is ``None`` whenever the trained/loaded
        vocab lacks ``[PAD]`` (custom ``special_tokens``, or a foreign
        ``tokenizer.json``). Callers fall back with ``getattr(tokenizer,
        "pad_token_id", 0)`` — but the attribute *exists* as None, so the
        default never fires and padding builds ``[None] * n``, crashing
        ``torch.tensor`` in ``batch_generate`` / ``TextDataset`` (RIL
        ISS-155). Fall back to the UNK token's id then the documented ``0``,
        so the property never returns ``None`` and never collides with
        content.

        Both lookups use explicit ``is not None`` checks, NOT an ``or``
        chain: ``[PAD]`` (or ``<unk>``) is legitimately vocab id ``0`` in
        some tokenizers, and ``0 or ...`` falls through to the WRONG token —
        padding every sample with the real content token ``<unk>`` instead of
        ``[PAD]`` (silent data corruption; deep-dive finding).
        """
        pad_id = self.tokenizer.token_to_id("[PAD]")
        if pad_id is not None:
            return pad_id
        unk_id = self.tokenizer.token_to_id(DEFAULT_UNK_TOKEN)
        return unk_id if unk_id is not None else 0
