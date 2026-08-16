class SimpleCharacterTokenizer:
    """
    A simple character-level tokenizer.

    This tokenizer builds a vocabulary from a given corpus and provides methods
    to encode text into a sequence of integer tokens and decode a sequence of
    tokens back into text.
    """

    pad_char: str = "<PAD>"
    eos_char: str = "<EOS>"
    bos_char: str = "<BOS>"

    def __init__(self, corpus: list[str]):
        """
        Initializes the SimpleCharacterTokenizer.

        Args:
            corpus (list[str]): A list of strings from which to build the vocabulary.
                                The vocabulary will consist of all unique characters
                                present in the corpus.
        """
        if not isinstance(corpus, list):
            raise TypeError("Corpus must be a list of strings.")
        if not all(isinstance(s, str) for s in corpus):
            raise TypeError("All items in the corpus must be strings.")

        # Join all strings in the corpus, then find unique characters
        unique_chars: set[str] = set("".join(corpus))
        self.chars: list[str] = sorted(unique_chars)  # Sort for consistent mapping

        self.stoi: dict[str, int] = {char: i for i, char in enumerate(self.chars)}
        self.itos: dict[int, str] = dict(enumerate(self.chars))
        self.vocab_size: int = len(self.chars)

        # Add PAD token
        if self.pad_char not in self.stoi:
            pad_token_id = self.vocab_size
            self.stoi[self.pad_char] = pad_token_id
            self.itos[pad_token_id] = self.pad_char
            self.chars.append(self.pad_char)  # Add to the list of characters
            self.vocab_size += 1
        else:
            # If PAD char was part of the corpus, use its existing ID
            pad_token_id = self.stoi[self.pad_char]

        self.pad_token_id: int = pad_token_id

        # ``<EOS>`` / ``<BOS>`` markers, declared alongside ``<PAD>`` by the
        # shared default corpus (``DEFAULT_SIMPLE_CORPUS``) and by
        # ``TokenizerFactory.from_dataset_text``. They are registered as real
        # multi-char special tokens *only when the corpus lists them*: without
        # this, the markers were flattened into their constituent plain
        # characters and ``eos_token_id`` stayed ``None``, so eval generations
        # (LMTask / lm_eval generate_until) never stopped on the model's EOS
        # (RIL ISS-152). Corpora that don't declare them keep the exact same
        # vocab layout as before.
        self._bos_token_id: int | None = None
        self._eos_token_id: int | None = None
        for marker, attr in ((self.eos_char, "_eos_token_id"), (self.bos_char, "_bos_token_id")):
            if marker in corpus and marker not in self.stoi:
                special_id = self.vocab_size
                self.stoi[marker] = special_id
                self.itos[special_id] = marker
                self.chars.append(marker)
                self.vocab_size += 1
                setattr(self, attr, special_id)

    @property
    def bos_token_id(self) -> int | None:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    def encode(self, text: str) -> list[int]:
        """
        Encodes a string of text into a list of integer tokens.

        Args:
            text (str): The input string to encode.

        Returns:
            list[int]: A list of integer tokens representing the input text.

        Raises:
            KeyError: If the text contains characters not present in the
                      tokenizer's vocabulary (i.e., not found in the
                      initial corpus).
        """
        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")

        # A special marker encoded verbatim maps to its single token id (the
        # pad precedent). Without this, ``encode("<EOS>")`` flattened to the
        # char ids of '<','E','O','S','>' even when the tokenizer had
        # registered the marker as a special (RIL ISS-152).
        if text in (self.pad_char, self.eos_char, self.bos_char):
            return [self.stoi[text]]

        tokens: list[int] = []
        for char in text:
            try:
                tokens.append(self.stoi[char])
            except KeyError:
                raise KeyError(
                    f"Character '{char}' not found in tokenizer vocabulary. "
                    "Only characters present in the initial corpus can be encoded."
                )
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """
        Decodes a list of integer tokens back into a string of text.

        Args:
            tokens (list[int]): A list of integer tokens to decode.

        Returns:
            str: The decoded string.

        Raises:
            KeyError: If the list contains token IDs not present in the
                      tokenizer's vocabulary.
        """
        if not isinstance(tokens, list):
            raise TypeError("Input tokens must be a list of integers.")
        if not all(isinstance(token, int) for token in tokens):
            raise TypeError("All items in the tokens list must be integers.")

        text_chars: list[str] = []
        for token in tokens:
            try:
                text_chars.append(self.itos[token])
            except KeyError:
                raise KeyError(
                    f"Token ID '{token}' not found in tokenizer vocabulary. "
                    "Only token IDs derived from the initial corpus can be decoded."
                )
        return "".join(text_chars)
