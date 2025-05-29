import pytest

from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer

# Test Constants
SAMPLE_CORPUS_BASIC = ["hello world", "pytest testing"]
# Expected sorted unique characters from SAMPLE_CORPUS_BASIC:
# ' ' -> 0, 'd' -> 1, 'e' -> 2, 'g' -> 3, 'h' -> 4, 'i' -> 5, 'l' -> 6, 'n' -> 7,
# 'o' -> 8, 'p' -> 9, 'r' -> 10, 's' -> 11, 't' -> 12, 'w' -> 13, 'y' -> 14
# Vocab size = 15 (before PAD)
INITIAL_EXPECTED_CHARS_BASIC = sorted(list(set("".join(SAMPLE_CORPUS_BASIC))))


SAMPLE_CORPUS_EXTENDED = ["hello world", "你好 世界"]  # Includes non-ASCII
INITIAL_EXPECTED_CHARS_EXTENDED = sorted(list(set("".join(SAMPLE_CORPUS_EXTENDED))))


@pytest.fixture
def basic_tokenizer():
    """Tokenizer based on SAMPLE_CORPUS_BASIC."""
    return SimpleCharacterTokenizer(SAMPLE_CORPUS_BASIC)


@pytest.fixture
def extended_tokenizer():
    """Tokenizer based on SAMPLE_CORPUS_EXTENDED."""
    return SimpleCharacterTokenizer(SAMPLE_CORPUS_EXTENDED)


class TestSimpleCharacterTokenizerInitialization:
    def test_pad_token_attributes(self, basic_tokenizer):
        assert hasattr(basic_tokenizer, "pad_char")
        assert hasattr(basic_tokenizer, "pad_token_id")
        assert basic_tokenizer.pad_char == "<PAD>"
        assert isinstance(basic_tokenizer.pad_token_id, int)
        assert basic_tokenizer.pad_char in basic_tokenizer.stoi
        assert basic_tokenizer.itos[basic_tokenizer.pad_token_id] == basic_tokenizer.pad_char
        assert basic_tokenizer.pad_char in basic_tokenizer.chars

    def test_vocab_creation_basic_with_pad(self, basic_tokenizer):
        # Initial chars + PAD token
        expected_vocab_size = len(INITIAL_EXPECTED_CHARS_BASIC) + 1
        assert basic_tokenizer.vocab_size == expected_vocab_size
        assert len(basic_tokenizer.chars) == expected_vocab_size
        assert len(basic_tokenizer.stoi) == expected_vocab_size
        assert len(basic_tokenizer.itos) == expected_vocab_size

        # Check initial chars are still there and correctly mapped
        for i, char_code in enumerate(INITIAL_EXPECTED_CHARS_BASIC):
            # The exact index might shift if PAD is not appended last or if corpus contained PAD.
            # The current implementation appends PAD if not present.
            assert char_code in basic_tokenizer.stoi
            original_char_idx = basic_tokenizer.stoi[char_code]
            assert basic_tokenizer.itos[original_char_idx] == char_code

        # Verify PAD token specifics
        assert basic_tokenizer.stoi[basic_tokenizer.pad_char] == basic_tokenizer.pad_token_id
        assert basic_tokenizer.itos[basic_tokenizer.pad_token_id] == basic_tokenizer.pad_char
        # Current implementation appends PAD, so its ID is len(initial_corpus_chars)
        assert basic_tokenizer.pad_token_id == len(INITIAL_EXPECTED_CHARS_BASIC)

    def test_vocab_creation_extended_with_pad(self, extended_tokenizer):
        expected_vocab_size = len(INITIAL_EXPECTED_CHARS_EXTENDED) + 1
        assert extended_tokenizer.vocab_size == expected_vocab_size
        assert extended_tokenizer.pad_char in extended_tokenizer.stoi
        assert extended_tokenizer.itos[extended_tokenizer.pad_token_id] == extended_tokenizer.pad_char

    def test_empty_corpus_with_pad(self):
        tokenizer = SimpleCharacterTokenizer([])
        # Vocab should only contain PAD token
        assert tokenizer.chars == [tokenizer.pad_char]
        assert tokenizer.vocab_size == 1
        assert tokenizer.stoi == {tokenizer.pad_char: 0}
        assert tokenizer.itos == {0: tokenizer.pad_char}
        assert tokenizer.pad_token_id == 0

    def test_corpus_with_empty_string_with_pad(self):
        tokenizer = SimpleCharacterTokenizer(["abc", "", "de"])
        initial_corpus_chars = sorted(list("abcde"))
        expected_vocab_size = len(initial_corpus_chars) + 1

        assert tokenizer.vocab_size == expected_vocab_size
        assert tokenizer.pad_char in tokenizer.stoi
        for char_code in initial_corpus_chars:
            assert char_code in tokenizer.stoi

    def test_corpus_containing_pad_char_string(self):
        # Test if corpus itself contains the string "<PAD>"
        corpus_with_pad_str = ["hello", "<PAD>world"]
        tokenizer = SimpleCharacterTokenizer(corpus_with_pad_str)

        # The tokenizer's self.pad_char is "<PAD>".
        # If corpus contains "<PAD>", it's treated as a regular character initially.
        # Then, the PAD token logic runs.
        # Current logic: if self.pad_char is in self.stoi (from corpus), self.pad_token_id takes that ID.
        # Vocab size should not double-count it.

        unique_single_chars_from_corpus = sorted(list(set("".join(corpus_with_pad_str))))
        # Assert that the individual characters from "<PAD>world" are in the tokenizer's single char mapping
        for char_in_pad_str_literal in "<PAD>":  # i.e. '<', 'P', 'A', 'D', '>'
            assert char_in_pad_str_literal in tokenizer.stoi
            assert char_in_pad_str_literal in unique_single_chars_from_corpus

        # Assert that the special PAD token "<PAD>" (as a whole string) is also in stoi and has the correct ID
        assert tokenizer.pad_char in tokenizer.stoi  # Checks for the string "<PAD>"
        assert tokenizer.stoi[tokenizer.pad_char] == tokenizer.pad_token_id

        # Vocab size should be number of unique single characters from corpus + 1 (for the special "<PAD>" token)
        # This is because "<PAD>" as a string is treated as a distinct token from its constituent characters '<', 'P', etc.
        # if the tokenizer's pad_char string itself ("<PAD>") was not already formed by the unique_chars set.
        # In the current SimpleCharacterTokenizer, unique_chars are single characters.
        # The pad_char "<PAD>" is added separately if not identical to one of the single chars.
        assert tokenizer.vocab_size == len(unique_single_chars_from_corpus) + 1

        # tokenizer.chars should contain all unique single characters and the special "<PAD>" string.
        # The order depends on the implementation (sorted single chars, then PAD usually)
        assert tokenizer.pad_char in tokenizer.chars
        for char_s in unique_single_chars_from_corpus:
            assert char_s in tokenizer.chars
        assert len(tokenizer.chars) == tokenizer.vocab_size

    def test_corpus_type_errors(self):
        with pytest.raises(TypeError, match="Corpus must be a list of strings"):
            SimpleCharacterTokenizer("not a list")  # type: ignore
        with pytest.raises(TypeError, match="All items in the corpus must be strings"):
            SimpleCharacterTokenizer(["a", "b", 123])  # type: ignore


class TestSimpleCharacterTokenizerEncoding:
    def test_encode_basic(self, basic_tokenizer):
        text = "hello"
        # h, e, l, l, o indices from INITIAL_EXPECTED_CHARS_BASIC
        # 'h' = INITIAL_EXPECTED_CHARS_BASIC.index('h'), etc.
        # This test needs to use the actual stoi values from the tokenizer fixture
        # as PAD token might shift indices if it's not appended last.
        # Current implementation appends PAD, so original indices are preserved.
        expected_tokens = [
            basic_tokenizer.stoi.get("h"),  # Use .get for robustness if char is missing, though test assumes it's there
            basic_tokenizer.stoi.get("e"),
            basic_tokenizer.stoi.get("l"),
            basic_tokenizer.stoi.get("l"),
            basic_tokenizer.stoi.get("o"),
        ]
        # Remove None if any char was unexpectedly missing (shouldn't happen for this test)
        expected_tokens = [t for t in expected_tokens if t is not None]
        assert basic_tokenizer.encode(text) == expected_tokens

    def test_encode_extended(self, extended_tokenizer):
        text = "你好"
        expected_tokens = [extended_tokenizer.stoi.get("你"), extended_tokenizer.stoi.get("好")]
        expected_tokens = [t for t in expected_tokens if t is not None]
        assert extended_tokenizer.encode(text) == expected_tokens

    def test_encode_empty_string(self, basic_tokenizer):
        assert basic_tokenizer.encode("") == []

    def test_encode_pad_char_itself(self, basic_tokenizer):
        # The <PAD> string itself is now a character in the vocab.
        assert basic_tokenizer.encode(basic_tokenizer.pad_char) == [basic_tokenizer.pad_token_id]

    def test_encode_unknown_chars(self, basic_tokenizer):
        with pytest.raises(KeyError, match="Character 'z' not found"):
            basic_tokenizer.encode("helloz")  # 'z' is not in SAMPLE_CORPUS_BASIC's initial chars

    def test_encode_type_error(self, basic_tokenizer):
        with pytest.raises(TypeError, match="Input text must be a string"):
            basic_tokenizer.encode(123)  # type: ignore


class TestSimpleCharacterTokenizerDecoding:
    def test_decode_basic(self, basic_tokenizer):
        # Same logic as test_encode_basic for token list construction
        tokens = [
            basic_tokenizer.stoi.get("h"),
            basic_tokenizer.stoi.get("e"),
            basic_tokenizer.stoi.get("l"),
            basic_tokenizer.stoi.get("l"),
            basic_tokenizer.stoi.get("o"),
        ]
        tokens = [t for t in tokens if t is not None]
        assert basic_tokenizer.decode(tokens) == "hello"

    def test_decode_extended(self, extended_tokenizer):
        tokens = [extended_tokenizer.stoi.get("你"), extended_tokenizer.stoi.get("好")]
        tokens = [t for t in tokens if t is not None]
        assert extended_tokenizer.decode(tokens) == "你好"

    def test_decode_pad_token_id(self, basic_tokenizer):
        assert basic_tokenizer.decode([basic_tokenizer.pad_token_id]) == basic_tokenizer.pad_char

    def test_decode_empty_list(self, basic_tokenizer):
        assert basic_tokenizer.decode([]) == ""

    def test_decode_unknown_tokens(self, basic_tokenizer):
        # vocab_size already includes PAD, so any ID >= vocab_size is unknown
        unknown_token_id = basic_tokenizer.vocab_size
        with pytest.raises(KeyError, match=f"Token ID '{unknown_token_id}' not found"):
            basic_tokenizer.decode([basic_tokenizer.stoi.get("h", 0), unknown_token_id])

    def test_decode_type_errors(self, basic_tokenizer):
        with pytest.raises(TypeError, match="Input tokens must be a list of integers"):
            basic_tokenizer.decode("not a list")  # type: ignore
        with pytest.raises(TypeError, match="All items in the tokens list must be integers"):
            basic_tokenizer.decode([1, 2, "c"])  # type: ignore


class TestSimpleCharacterTokenizerEndToEnd:
    @pytest.mark.parametrize("text_to_test", ["hello world", "pytest", "testing", " ", ""])
    def test_encode_decode_identity_basic(self, basic_tokenizer, text_to_test):
        # Ensure text_to_test only contains chars from the vocab
        # Ensure text_to_test only contains chars from the vocab (excluding PAD for this type of test)
        for char_code in text_to_test:
            if char_code == basic_tokenizer.pad_char:  # Skip if text is PAD char itself
                pytest.skip("Skipping direct PAD char string for this identity test, tested elsewhere.")
            if char_code not in basic_tokenizer.stoi:
                pytest.skip(f"Character '{char_code}' not in basic_tokenizer vocab for identity test.")

        encoded = basic_tokenizer.encode(text_to_test)
        decoded = basic_tokenizer.decode(encoded)
        assert decoded == text_to_test

    @pytest.mark.parametrize("text_to_test", ["hello 世界", "你好", " "])
    def test_encode_decode_identity_extended(self, extended_tokenizer, text_to_test):
        for char_code in text_to_test:
            if char_code == extended_tokenizer.pad_char:
                pytest.skip("Skipping direct PAD char string for this identity test.")
            if char_code not in extended_tokenizer.stoi:
                pytest.skip(f"Character '{char_code}' not in extended_tokenizer vocab for identity test.")

        encoded = extended_tokenizer.encode(text_to_test)
        decoded = extended_tokenizer.decode(encoded)
        assert decoded == text_to_test

    def test_encode_decode_identity_pad_char(self, basic_tokenizer):
        """Test encode/decode of the PAD character string itself."""
        encoded = basic_tokenizer.encode(basic_tokenizer.pad_char)
        decoded = basic_tokenizer.decode(encoded)
        assert decoded == basic_tokenizer.pad_char
        assert encoded == [basic_tokenizer.pad_token_id]

    def test_encode_decode_identity_empty_corpus_tokenizer(self):
        # For a tokenizer from an empty corpus, only PAD and empty string can be processed.
        tokenizer = SimpleCharacterTokenizer([])  # Contains only PAD after init

        # Test empty string
        text_empty = ""
        encoded_empty = tokenizer.encode(text_empty)
        decoded_empty = tokenizer.decode(encoded_empty)
        assert decoded_empty == text_empty

        # Test PAD char itself
        text_pad = tokenizer.pad_char
        encoded_pad = tokenizer.encode(text_pad)
        decoded_pad = tokenizer.decode(encoded_pad)
        assert decoded_pad == text_pad
        assert encoded_pad == [tokenizer.pad_token_id]

        with pytest.raises(KeyError):  # Any other non-empty string will fail
            tokenizer.encode("a")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
