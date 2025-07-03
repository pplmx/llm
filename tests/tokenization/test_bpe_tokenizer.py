import pytest
import os
import json
from tempfile import TemporaryDirectory

from llm.tokenization.bpe_tokenizer import BPETokenizer

@pytest.fixture
def sample_corpus():
    return [
        "hello world",
        "hello there",
        "this is a test",
        "another test example for the tokenizer",
        "hello again world to test merges",
        "low lower lowest",
        "hugging an apple", # to create 'hu', 'gg', 'in', 'g ', 'ap', 'pl', 'le' potentially
    ]

@pytest.fixture
def simple_corpus():
    return ["aaab", "aaac"] # For predictable merges

class TestBPETokenizer:
    def test_initialization(self):
        tokenizer = BPETokenizer(vocab_size=100)
        assert tokenizer.vocab_size == 100
        assert tokenizer.pad_token == "[PAD]"
        assert tokenizer.unk_token == "[UNK]"
        assert tokenizer.eow_token == "</w>"
        assert tokenizer.pad_token_id is not None
        assert tokenizer.unk_token_id is not None
        assert tokenizer.vocab[tokenizer.pad_token] == tokenizer.pad_token_id
        assert tokenizer.vocab[tokenizer.unk_token] == tokenizer.unk_token_id
        assert len(tokenizer.vocab) == 2 # Only PAD and UNK initially

    def test_initialization_invalid_vocab_size(self):
        with pytest.raises(ValueError, match="Vocabulary size must be positive"):
            BPETokenizer(vocab_size=0)
        with pytest.raises(ValueError, match="Vocabulary size must be positive"):
            BPETokenizer(vocab_size=-10)

    def test_train_simple_corpus_and_merges(self, simple_corpus):
        # Test with a very small vocab size to force specific merges
        # Corpus: ["aaab", "aaac"]
        # Initial chars + </w>: a, b, c, </w>, [PAD], [UNK] (6)
        # Target vocab size: 8 (allows for 2 merges)
        # Word freqs: aaab</w>: 1, aaac</w>: 1
        # Pretokenized: (a,a,a,b,</w>), (a,a,a,c,</w>)
        # Pair freqs: (a,a): 4, (a,b): 1, (b,</w>): 1, (a,c): 1, (c,</w>):1
        # 1st merge: (a,a) -> "aa"
        #   New vocab: "aa". Vocab size: 7. Merges: [(('a','a'), "aa")]
        #   Corpus becomes: (aa,a,b,</w>), (aa,a,c,</w>)
        # Pair freqs: (aa,a):2, (a,b):1, (b,</w>):1, (a,c):1, (c,</w>):1
        # 2nd merge: (aa,a) -> "aaa"
        #   New vocab: "aaa". Vocab size: 8. Merges: [(('a','a'), "aa"), (('aa','a'), "aaa")]
        #   Corpus becomes: (aaa,b,</w>), (aaa,c,</w>)
        tokenizer = BPETokenizer(vocab_size=8) # PAD, UNK, a, b, c, </w> (6) + 2 merges
        tokenizer.train(simple_corpus, verbose=False)

        assert tokenizer.effective_vocab_size == 8
        assert "aa" in tokenizer.vocab
        assert "aaa" in tokenizer.vocab
        assert (("a", "a"), "aa") in tokenizer.merges
        assert (("aa", "a"), "aaa") in tokenizer.merges

        # Check if merges are ordered
        merge_strings = [m[1] for m in tokenizer.merges]
        assert merge_strings == ["aa", "aaa"]


    def test_train_on_sample_corpus(self, sample_corpus):
        tokenizer = BPETokenizer(vocab_size=60) # Expecting initial chars + some merges
        tokenizer.train(sample_corpus, verbose=False)

        initial_chars = set()
        for text in sample_corpus:
            for char_token in list(text.lower()) + ["</w>"]: # Mimic _pre_tokenize_word and _get_word_frequencies
                 if char_token.isspace(): # In training, spaces are delimiters, not tokens
                     for word_char in char_token.split(): # Should not happen with current regex
                         initial_chars.add(word_char)
                 elif char_token == "</w>":
                     initial_chars.add(char_token)
                 else: # from word_pattern = re.compile(r"\w+|[^\s\w]")
                     for part in re.findall(r"\w+|[^\s\w]", char_token):
                         for c in part:
                            initial_chars.add(c)


        # Add special tokens that are always there
        initial_chars.add("[PAD]")
        initial_chars.add("[UNK]")

        # The actual number of unique characters + eow_token from the corpus
        # plus PAD and UNK, form the base before merges.
        # The regex r"\w+|[^\s\w]" on "hugging an apple" gives 'h','u','g','g','i','n','g','a','n','a','p','p','l','e'
        # So spaces are not part of initial_chars from word_pattern.

        # Let's calculate initial vocab more directly based on train() logic
        temp_tokenizer_for_initial_vocab = BPETokenizer(vocab_size=200) # large enough
        word_freqs_orig = temp_tokenizer_for_initial_vocab._get_word_frequencies(sample_corpus)
        tokenized_word_freqs_init = {
            tuple(temp_tokenizer_for_initial_vocab._pre_tokenize_word(word)): freq
            for word, freq in word_freqs_orig.items()
        }
        base_char_vocab = set()
        for word_tuple in tokenized_word_freqs_init:
            for char_token in word_tuple:
                base_char_vocab.add(char_token)

        # Add PAD and UNK (which are added by default in __init__)
        expected_initial_vocab_count = len(base_char_vocab) + 2 # PAD, UNK

        assert tokenizer.effective_vocab_size <= 60
        assert tokenizer.effective_vocab_size >= expected_initial_vocab_count if expected_initial_vocab_count <= 60 else 60
        assert len(tokenizer.merges) == (tokenizer.effective_vocab_size - expected_initial_vocab_count) if tokenizer.effective_vocab_size > expected_initial_vocab_count else 0


    def test_train_empty_corpus(self):
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train([]) # Should not raise error, just result in minimal vocab
        assert tokenizer.effective_vocab_size == 2 # Only PAD and UNK
        assert not tokenizer.merges

    def test_train_vocab_size_too_small(self, sample_corpus):
        tokenizer = BPETokenizer(vocab_size=10) # Smaller than initial chars
        tokenizer.train(sample_corpus)
        # Vocab will contain initial chars up to vocab_size, plus PAD/UNK if they fit
        # This behavior: vocab is filled by sorted chars, then special tokens if space.
        # Current train(): adds PAD/UNK first, then sorted base_chars. If target is too small,
        # it might not add all base_chars.
        # Let's re-evaluate: _add_special_tokens_to_vocab runs first.
        # Then train adds sorted base_vocab. If len(self.vocab) from special + base > self.vocab_size,
        # no merges are done.
        # The effective_vocab_size will be min(initial_chars_count + special_tokens, target_vocab_size)
        # if no merges are possible / needed.
        # If target_vocab_size is 10, PAD, UNK take 2. 8 left for chars.

        # Calculate initial char count
        base_chars = set()
        word_freqs_orig = tokenizer._get_word_frequencies(sample_corpus)
        tokenized_word_freqs_init = {
            tuple(tokenizer._pre_tokenize_word(word)): freq
            for word, freq in word_freqs_orig.items()
        }
        for word_tuple in tokenized_word_freqs_init:
            for char_token in word_tuple:
                base_chars.add(char_token)

        # Expected size is min(target_vocab_size, count of unique chars + </w> + special tokens)
        # Special tokens (PAD, UNK) are added first = 2
        # Then sorted unique characters from corpus are added.
        # If target_vocab_size = 10, PAD, UNK make it 2. Max 8 more chars.
        expected_size = 2 + min(len(base_chars), 10 - 2)

        assert tokenizer.effective_vocab_size == expected_size
        assert not tokenizer.merges # No merges should happen


    def test_encode_decode_simple(self, simple_corpus):
        tokenizer = BPETokenizer(vocab_size=8) # Allows for 'aa' and 'aaa'
        tokenizer.train(simple_corpus)

        text = "aaab"
        encoded = tokenizer.encode(text) # Should be [id("aaa"), id("b"), id("</w>")]

        # Expected: 'aaa', 'b', '</w>'
        aaa_id = tokenizer.vocab.get("aaa")
        b_id = tokenizer.vocab.get("b")
        eow_id = tokenizer.vocab.get("</w>")

        assert aaa_id is not None
        assert b_id is not None
        assert eow_id is not None

        assert encoded == [aaa_id, b_id, eow_id]

        decoded = tokenizer.decode(encoded)
        assert decoded == "aaab" # after lowercasing and eow handling

        text2 = "aaac"
        encoded2 = tokenizer.encode(text2)
        c_id = tokenizer.vocab.get("c")
        assert c_id is not None
        assert encoded2 == [aaa_id, c_id, eow_id]
        decoded2 = tokenizer.decode(encoded2)
        assert decoded2 == "aaac"

    def test_encode_unknown_chars(self, simple_corpus):
        tokenizer = BPETokenizer(vocab_size=8)
        tokenizer.train(simple_corpus) # Vocab: PAD, UNK, a,b,c,</w>, aa, aaa

        text = "xyz" # x, y, z are unknown
        encoded = tokenizer.encode(text)

        unk_id = tokenizer.unk_token_id
        eow_id = tokenizer.vocab.get("</w>")
        assert unk_id is not None
        assert eow_id is not None

        # Expected: x -> unk, y -> unk, z -> unk, then eow
        assert encoded == [unk_id, unk_id, unk_id, eow_id]
        decoded = tokenizer.decode(encoded)
        assert decoded == f"{tokenizer.unk_token}{tokenizer.unk_token}{tokenizer.unk_token}" # or how UNK is handled in decode

    def test_encode_empty_string(self):
        tokenizer = BPETokenizer(vocab_size=10)
        tokenizer.train(["a b c"])
        encoded = tokenizer.encode("")
        assert encoded == []
        decoded = tokenizer.decode([])
        assert decoded == ""

    def test_decode_handles_eow_correctly(self, sample_corpus):
        tokenizer = BPETokenizer(vocab_size=60)
        tokenizer.train(sample_corpus)

        text = "hello world"
        encoded = tokenizer.encode(text) # Will include eow for "hello" and "world"
        decoded = tokenizer.decode(encoded)
        assert decoded == "hello world"

        text_single = "test"
        encoded_single = tokenizer.encode(text_single)
        decoded_single = tokenizer.decode(encoded_single)
        assert decoded_single == "test"

        # Test multiple words and punctuation (though punc handling is basic)
        text_punc = "hello. world!"
        encoded_punc = tokenizer.encode(text_punc)
        decoded_punc = tokenizer.decode(encoded_punc)
        assert decoded_punc == "hello . world !" # current regex splits '.' and '!'

    def test_save_and_load_vocab(self, sample_corpus):
        original_tokenizer = BPETokenizer(vocab_size=70)
        original_tokenizer.train(sample_corpus, verbose=False)

        text_to_test = "hello this is a tokenizer test with merges"
        original_encoded = original_tokenizer.encode(text_to_test)
        original_decoded = original_tokenizer.decode(original_encoded)

        with TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "bpe_vocab.json")
            original_tokenizer.save_vocab(filepath)

            assert os.path.exists(filepath)

            loaded_tokenizer = BPETokenizer.load_vocab(filepath)

            # Compare attributes
            assert loaded_tokenizer.vocab_size == original_tokenizer.vocab_size
            assert loaded_tokenizer.pad_token == original_tokenizer.pad_token
            assert loaded_tokenizer.unk_token == original_tokenizer.unk_token
            assert loaded_tokenizer.eow_token == original_tokenizer.eow_token
            assert loaded_tokenizer.vocab == original_tokenizer.vocab
            assert loaded_tokenizer.merges == original_tokenizer.merges
            assert loaded_tokenizer.ids_to_tokens == original_tokenizer.ids_to_tokens
            assert loaded_tokenizer.pad_token_id == original_tokenizer.pad_token_id
            assert loaded_tokenizer.unk_token_id == original_tokenizer.unk_token_id
            assert loaded_tokenizer.effective_vocab_size == original_tokenizer.effective_vocab_size

            # Test encoding/decoding with loaded tokenizer
            loaded_encoded = loaded_tokenizer.encode(text_to_test)
            loaded_decoded = loaded_tokenizer.decode(loaded_encoded)

            assert loaded_encoded == original_encoded
            assert loaded_decoded == original_decoded

    def test_padding_and_unknown_token_ids_in_decode(self):
        tokenizer = BPETokenizer(vocab_size=10)
        tokenizer.train(["a"])
        pad_id = tokenizer.pad_token_id
        unk_id = tokenizer.unk_token_id
        a_id = tokenizer.vocab['a']
        eow_id = tokenizer.vocab['</w>']

        # Decoding with PAD should skip PAD
        decoded_with_pad = tokenizer.decode([a_id, pad_id, eow_id, pad_id])
        assert decoded_with_pad == "a"

        # Decoding with UNK should render UNK token
        # Create a fake ID that is not in ids_to_tokens for a robust test of this path
        max_valid_id = max(tokenizer.ids_to_tokens.keys())
        fake_unknown_id = max_valid_id + 10

        decoded_with_unk = tokenizer.decode([a_id, eow_id, unk_id, fake_unknown_id])
        # Expects "a <unk> <?>", where <?> is if unk_token is None or ID is truly out of range
        # Current decode for totally unknown ID gives "<?>" if unk_token is None,
        # or unk_token string if ID is simply not in ids_to_tokens but unk_token is defined.
        # Here, unk_id should map to unk_token string. fake_unknown_id will map to "<?>" if unk_token is None.
        # Since unk_token is "[UNK]", it should be:
        expected_decode_unk = f"a {tokenizer.unk_token} {tokenizer.unk_token}" # Both unk_id and fake_unknown_id should resolve to unk_token string
        # Correction: decode() maps unknown IDs to self.unk_token string if defined.
        # So, both unk_id and fake_unknown_id (if not pad_id) will be self.unk_token
        assert decoded_with_unk == f"a {tokenizer.unk_token} {tokenizer.unk_token}"


    def test_consistency_after_load_for_merges(self):
        # Test a specific case where merges are crucial
        corpus = [" বারবার ", " ফুটবল "] # Bengali words, "barbar", "football"
        # Target: learn "বার" (bar), then "বারবার" (barbar)
        # Initial chars: ব,া,র,ฟ,ุ,ট,ব,ল, </w>, PAD, UNK
        # Target vocab for this test could be small, e.g., 15 to see merges
        original_tokenizer = BPETokenizer(vocab_size=30)
        original_tokenizer.train(corpus, verbose=False)

        test_word = "বারবার" # barbar
        original_encoded = original_tokenizer.encode(test_word)

        with TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "bpe_complex_merges.json")
            original_tokenizer.save_vocab(filepath)
            loaded_tokenizer = BPETokenizer.load_vocab(filepath)

        loaded_encoded = loaded_tokenizer.encode(test_word)
        assert original_encoded == loaded_encoded, "Encoding mismatch after load for complex merges"

        assert original_tokenizer.decode(original_encoded) == loaded_tokenizer.decode(loaded_encoded)
        assert loaded_tokenizer.decode(loaded_encoded) == test_word.lower() # as corpus is lowercased

    def test_word_pattern_impact(self):
        # Default pattern: r"\w+|[^\s\w]"
        tokenizer = BPETokenizer(vocab_size=30)
        corpus = ["word1 word2.word3"]
        tokenizer.train(corpus)
        # Expected tokens from "word2.word3": "word2", ".", "word3"
        # Each gets </w>
        # "word2</w>", ".</w>", "word3</w>"
        encoded = tokenizer.encode("word2.word3")
        decoded = tokenizer.decode(encoded)
        assert decoded == "word2 . word3" # space before . and after . because of eow

        # Example from SimpleCharacterTokenizer that might be an issue if not handled
        text_with_special_char_word = "hello <PAD> world" # <PAD> is a word here
        encoded_special = tokenizer.encode(text_with_special_char_word)
        # If '<PAD>' is treated as a sequence of chars '<', 'P', 'A', 'D', '>'
        # vs. a single token.
        # Our BPE splits by word_pattern first. So "<PAD>" is one "word".
        # Then it becomes ['<', 'P', 'A', 'D', '>', '</w>']
        # Unless '<PAD>' itself becomes a merged token or is an initial char.
        # It is a special token, but not part of char set unless in corpus.
        # The BPETokenizer adds self.pad_token to self.vocab with a specific ID.
        # encode() uses self.vocab.get(token, self.unk_token_id).
        # _tokenize_word_with_merges result in list of strings.
        # If one of these strings is exactly self.pad_token, it will get pad_token_id.
        # This is unlikely unless a merge results in "[PAD]".

        # Let's test if a special token string in input text gets tokenized as itself or broken down
        # if it's not part of the trained merges/vocab (beyond the special token list).
        tokenizer_pad_test = BPETokenizer(vocab_size=30, pad_token="[MYPAD]")
        tokenizer_pad_test.train(["some text"])

        # Case 1: The special token string appears in text
        text_containing_pad_str = "this is [MYPAD] text"
        encoded_pad_str = tokenizer_pad_test.encode(text_containing_pad_str)

        # Expected: "[MYPAD]" is a "word" by regex. Pretokenized to ['[', 'M', 'Y', 'P', 'A', 'D', ']', '</w>']
        # These chars will be tokenized, likely to UNK if not in base vocab and no merges make them.
        # It should NOT resolve to tokenizer_pad_test.pad_token_id unless a merge forms "[MYPAD]"
        # AND "[MYPAD]" was added to vocab during training.
        # This highlights that special tokens are for protocol, not necessarily for direct text matching if complex.

        # Let's simplify: what if a token from BPE merges looks like a special token?
        # e.g. if ('[', 'P') -> "[P", then ('[P', 'AD') -> "[PAD" etc.
        # This is fine, they are just strings in the vocab.
        # The `decode` method specifically checks `token == self.pad_token` to skip.

        # The main point is that `encode` splits by word_pattern.
        # If a word from `word_pattern` is exactly a special token string (e.g. "[PAD]"),
        # it becomes list_of_chars + eow, then BPE merges apply.
        # The final list of string tokens is looked up in self.vocab.
        # If "word" from word_pattern is "[PAD]", and "[PAD]" is in self.vocab (as a special token),
        # then list_of_chars + eow is ['[','P','A','D',']','</w>']. This will likely not merge to "[PAD]".
        # So it will be tokenized as individual (possibly UNK) chars.
        # This is generally the desired behavior: special tokens are for control, not literal text.
        # If a user *wants* "[PAD]" in text to be a single token, it must be learned by BPE or be a base char.

        # No direct assert here, more of a conceptual check of current behavior.
        pass
```
