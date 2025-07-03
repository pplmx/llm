from collections import defaultdict
import re

class BPETokenizer:
    def __init__(self, vocab_size: int, pad_token: str = "[PAD]", unk_token: str = "[UNK]", eow_token: str = "</w>"):
        """
        Initializes the BPE Tokenizer.

        Args:
            vocab_size (int): The target vocabulary size, including initial characters and learned merges.
            pad_token (str, optional): Special token for padding. Defaults to "[PAD]".
            unk_token (str, optional): Special token for unknown words. Defaults to "[UNK]".
            eow_token (str, optional): Special token to mark the end of a word. Defaults to "</w>".
        """
        if vocab_size <= 0:
            raise ValueError("Vocabulary size must be positive.")

        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eow_token = eow_token # End-of-word token

        # Vocabulary: maps tokens to integer IDs
        self.vocab: dict[str, int] = {}
        # Merges: stores the learned merge operations, ordered by learning
        # For simplicity, let's store them as a list of tuples: (pair_tuple, merged_token_str)
        self.merges: list[tuple[tuple[str, str], str]] = []
        # Inverse vocabulary: maps integer IDs to tokens
        self.ids_to_tokens: dict[int, str] = {}

        # Initialize vocab with special tokens if they are not None
        self._add_special_tokens_to_vocab()

    def _add_special_tokens_to_vocab(self):
        """Adds special tokens to the vocabulary if they are defined."""
        special_tokens = []
        if self.pad_token is not None:
            special_tokens.append(self.pad_token)
        if self.unk_token is not None:
            special_tokens.append(self.unk_token)
        # Note: eow_token is part of the merging process, not treated as a standalone special token
        # in the same way as PAD or UNK for direct addition here, but it will be part of the vocab.

        for token in special_tokens:
            if token not in self.vocab:
                token_id = len(self.vocab)
                self.vocab[token] = token_id
                self.ids_to_tokens[token_id] = token

    @property
    def pad_token_id(self) -> int | None:
        return self.vocab.get(self.pad_token)

    @property
    def unk_token_id(self) -> int | None:
        return self.vocab.get(self.unk_token)

    @property
    def effective_vocab_size(self) -> int:
        """Returns the current actual size of the vocabulary mapping."""
        return len(self.vocab)

    def _get_word_frequencies(self, corpus: list[str]) -> dict[str, int]:
        """
        Calculates frequencies of words in the corpus.
        Words are typically space-separated tokens.
        Further splitting (e.g., by punctuation) might be needed for more robust BPE.
        For now, we'll use a simple regex to extract words.
        """
        word_freqs = defaultdict(int)
        # A simple regex to find sequences of alphabetic characters or standalone non-alphabetic, non-space characters.
        # This is a basic word definition and can be improved.
        word_pattern = re.compile(r"\w+|[^\s\w]")
        for text in corpus:
            words = word_pattern.findall(text.lower()) # Lowercasing for consistency
            for word in words:
                word_freqs[word] += 1
        return word_freqs

    def _pre_tokenize_word(self, word: str) -> list[str]:
        """
        Splits a word into characters and appends the end-of-word token.
        Example: "hello" -> ['h', 'e', 'l', 'l', 'o', '</w>']
        """
        return list(word) + [self.eow_token]

    def _get_stats(self, tokenized_word_freqs: dict[tuple[str, ...], int]) -> defaultdict[tuple[str, str], int]:
        """
        Calculates frequencies of adjacent pairs of symbols.
        Args:
            tokenized_word_freqs: A dictionary where keys are tuples of symbols (pre-tokenized words)
                                  and values are their frequencies.
                                  Example: {('h', 'e', 'l', 'l', 'o', '</w>'): 5, ...}
        Returns:
            A defaultdict mapping pairs of symbols to their frequencies.
        """
        pair_freqs = defaultdict(int)
        for word_tuple, freq in tokenized_word_freqs.items():
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i+1])
                pair_freqs[pair] += freq
        return pair_freqs

    def _merge_stats_and_update_word_freqs(
        self,
        pair_to_merge: tuple[str, str],
        tokenized_word_freqs: dict[tuple[str, ...], int]
    ) -> dict[tuple[str, ...], int]:
        """
        Merges the given pair in the keys of tokenized_word_freqs.
        The new merged token is simply pair_to_merge[0] + pair_to_merge[1].

        Args:
            pair_to_merge: The pair of symbols to merge (e.g., ('h', 'e')).
            tokenized_word_freqs: Word frequencies with tuple-tokenized words as keys.

        Returns:
            A new dictionary with updated tokenized_word_freqs.
        """
        new_tokenized_word_freqs = {}
        merged_token_str = "".join(pair_to_merge)

        for word_tuple, freq in tokenized_word_freqs.items():
            new_word_tuple_list = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i+1]) == pair_to_merge:
                    new_word_tuple_list.append(merged_token_str)
                    i += 2
                else:
                    new_word_tuple_list.append(word_tuple[i])
                    i += 1
            new_tokenized_word_freqs[tuple(new_word_tuple_list)] = freq
        return new_tokenized_word_freqs

    def train(self, corpus: list[str], verbose: bool = False):
        """
        Trains the BPE tokenizer from a corpus.

        Args:
            corpus (list[str]): A list of strings (sentences or documents) to train on.
            verbose (bool, optional): Whether to print progress. Defaults to False.
        """
        if not corpus:
            raise ValueError("Corpus cannot be empty.")

        # 1. Initialize vocabulary with base characters
        base_vocab = set()
        word_freqs_orig = self._get_word_frequencies(corpus)

        if not word_freqs_orig:
            print("Warning: No words found in the corpus. Tokenizer will not be trained effectively.")
            return

        # Create a working copy of word frequencies where keys are tuples of symbols
        # e.g., "hello" (freq 5) -> {('h','e','l','l','o','</w>'): 5}
        tokenized_word_freqs: dict[tuple[str,...], int] = {
            tuple(self._pre_tokenize_word(word)): freq
            for word, freq in word_freqs_orig.items()
        }

        for word_tuple in tokenized_word_freqs:
            for char_token in word_tuple:
                base_vocab.add(char_token)

        # Add base characters to vocab, excluding already added special tokens
        for char_token in sorted(list(base_vocab)): # sorted for deterministic order
            if char_token not in self.vocab:
                token_id = len(self.vocab)
                self.vocab[char_token] = token_id
                self.ids_to_tokens[token_id] = char_token

        if verbose:
            print(f"Initial vocabulary size: {len(self.vocab)}")
            # print(f"Initial vocab: {self.vocab}")

        # 2. Learn merges
        num_merges_needed = self.vocab_size - len(self.vocab)
        if num_merges_needed < 0 :
            print(f"Warning: Initial vocab size ({len(self.vocab)}) already exceeds target vocab_size ({self.vocab_size}). No merges will be learned.")
            num_merges_needed = 0
            # Consider if self.vocab should be trimmed or if this is acceptable.
            # For now, we'll proceed with the current vocab.

        for i in range(num_merges_needed):
            pair_freqs = self._get_stats(tokenized_word_freqs)

            if not pair_freqs:
                if verbose:
                    print("No more pairs to merge. Stopping training.")
                break # No more pairs to merge

            # Find the most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            best_pair_freq = pair_freqs[best_pair]

            if best_pair_freq == 0 : # Should not happen if pair_freqs is not empty
                 if verbose:
                    print("Most frequent pair has frequency 0. Stopping training.")
                 break

            # Merge the best pair
            merged_token_str = "".join(best_pair)
            tokenized_word_freqs = self._merge_stats_and_update_word_freqs(best_pair, tokenized_word_freqs)

            # Add new merged token to vocabulary and merges list
            if merged_token_str not in self.vocab: # Important check
                token_id = len(self.vocab)
                self.vocab[merged_token_str] = token_id
                self.ids_to_tokens[token_id] = merged_token_str
                self.merges.append((best_pair, merged_token_str))

                if verbose and (i % 100 == 0 or i == num_merges_needed -1):
                    print(f"Merge {i+1}/{num_merges_needed}: Merged {best_pair} -> {merged_token_str} (freq: {best_pair_freq}). Vocab size: {len(self.vocab)}")
            else:
                # This case (merged token already in vocab) implies we might have a sub-optimal merge order choice
                # or the vocab_size target is too small for the number of distinct characters and desired merges.
                # For simplicity, we'll record the merge rule but not add to vocab.
                # This could also mean we need to re-evaluate how num_merges_needed is calculated if tokens can be formed
                # that were already base characters (e.g. if "a" "b" -> "ab" and "ab" was a word initially)
                # However, our pre_tokenize_word splits everything, so this is less likely for char-level BPE.
                self.merges.append((best_pair, merged_token_str))
                if verbose and (i % 100 == 0 or i == num_merges_needed -1):
                     print(f"Merge {i+1}/{num_merges_needed}: Merged {best_pair} -> {merged_token_str} (freq: {best_pair_freq}). Token already in vocab. Vocab size: {len(self.vocab)}")


            # If vocab size target is reached prematurely due to existing tokens
            if len(self.vocab) >= self.vocab_size:
                if verbose:
                    print(f"Target vocabulary size {self.vocab_size} reached. Stopping training.")
                break

        if verbose:
            print(f"Training complete. Final vocabulary size: {len(self.vocab)}")
            print(f"Number of learned merge rules: {len(self.merges)}")


    def _tokenize_word_with_merges(self, word_tuple: list[str]) -> list[str]:
        """
        Applies learned BPE merges to a pre-tokenized word (list of characters + eow).
        """
        if not self.merges:
            return word_tuple

        current_word_symbols = list(word_tuple) # Make a mutable copy

        while True:
            applied_merge = False
            min_merge_idx = float('inf')
            best_pair_to_apply = None

            # Iterate through all learned merges to find the one that appears earliest
            # and has the lowest merge rank (index in self.merges)
            # This ensures merges are applied in the order they were learned.

            # Find all possible merges for the current symbols
            possible_merges_in_current_symbols = [] # Stores (merge_idx_in_self.merges, pair_idx_in_word, pair_tuple)

            for i in range(len(current_word_symbols) - 1):
                pair = (current_word_symbols[i], current_word_symbols[i+1])
                for merge_idx, (learned_pair, _) in enumerate(self.merges):
                    if pair == learned_pair:
                        possible_merges_in_current_symbols.append((merge_idx, i, pair))
                        break # Found the merge rule for this pair, move to next pair in word

            if not possible_merges_in_current_symbols:
                break # No learned merges can be applied

            # Select the merge that has the lowest merge_idx (was learned earliest)
            # If ties, pick the one that appears first in the word (lowest pair_idx_in_word)
            possible_merges_in_current_symbols.sort(key=lambda x: (x[0], x[1]))

            best_merge_idx, best_pair_start_idx, pair_to_apply = possible_merges_in_current_symbols[0]
            merged_token_str = self.merges[best_merge_idx][1] # Get the string representation of the merged token

            # Apply the selected merge
            new_symbols = []
            j = 0
            while j < len(current_word_symbols):
                if j == best_pair_start_idx:
                    new_symbols.append(merged_token_str)
                    j += 2 # Skip the two symbols that were merged
                    applied_merge = True
                else:
                    new_symbols.append(current_word_symbols[j])
                    j += 1
            current_word_symbols = new_symbols

            if not applied_merge: # Should not happen if possible_merges_in_current_symbols was not empty
                break

        return current_word_symbols


    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: # `add_special_tokens` placeholder for now
        """
        Tokenizes input text into a list of token IDs.
        `add_special_tokens` is not fully implemented yet (e.g. SOS/EOS).
        """
        token_ids = []

        # Basic word splitting, can be improved (e.g. to handle punctuation better with regex from training)
        word_pattern = re.compile(r"\w+|[^\s\w]") # Consistent with training
        words = word_pattern.findall(text.lower()) # Lowercasing for consistency

        for word in words:
            if not word: continue

            pre_tokenized_word_list = self._pre_tokenize_word(word)

            # Apply learned merges to this single word's symbols
            final_word_tokens = self._tokenize_word_with_merges(pre_tokenized_word_list)

            for token in final_word_tokens:
                token_id = self.vocab.get(token, self.unk_token_id)
                if token_id is None and self.unk_token is None: # Should not happen if unk_token is always set
                    raise ValueError(f"Token '{token}' not in vocab and no UNK token defined.")
                elif token_id is None: # Fallback, should be caught by self.unk_token_id
                     raise Exception("Logic error: unk_token_id should not be None if unk_token is defined")
                token_ids.append(token_id)

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Converts a list of token IDs back to a string.
        """
        tokens = []
        for token_id in token_ids:
            token = self.ids_to_tokens.get(token_id)
            if token is None:
                # This case implies an ID was passed that's not in ids_to_tokens
                # which could happen if unk_token_id was from a different tokenizer or vocab was corrupted
                # For robustness, one might map this to unk_token string if defined, or raise error/warning
                if self.unk_token is not None:
                    tokens.append(self.unk_token)
                else:
                    # Or raise an error: raise ValueError(f"Invalid token ID: {token_id}")
                    tokens.append("<?>") # Placeholder for unknown ID if no UNK token
            elif token == self.pad_token: # Optionally skip pad tokens in output
                continue
            else:
                tokens.append(token)

        # Basic detokenization: join tokens, then try to handle eow_token
        # This is a simplified detokenization. Real BPE detokenization can be more complex
        # especially with SentencePiece style tokenization that handles spaces differently.
        reconstructed_text = "".join(tokens)

        # Replace eow_token with a space, then clean up multiple spaces and strip.
        if self.eow_token:
            reconstructed_text = reconstructed_text.replace(self.eow_token, " ")

        reconstructed_text = re.sub(r"\s+", " ", reconstructed_text).strip()

        return reconstructed_text

    def save_vocab(self, filepath: str):
        """Saves the tokenizer's vocabulary and merges to a file."""
        import json
        # Ensure special tokens are part of the vocab being saved explicitly
        # if they were somehow not added or if IDs need to be preserved.
        # Current _add_special_tokens_to_vocab and train() should handle this.
        data = {
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "eow_token": self.eow_token,
            "vocab": self.vocab,
            "merges": self.merges # Storing merges is crucial for BPE
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer vocabulary and merges saved to {filepath}")

    @classmethod
    def load_vocab(cls, filepath: str) -> "BPETokenizer":
        """Loads the tokenizer's vocabulary and merges from a file."""
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"],
            pad_token=data.get("pad_token"), # Use .get for backward compatibility if a token was None
            unk_token=data.get("unk_token"),
            eow_token=data.get("eow_token", "</w>") # Default if not in older save
        )
        tokenizer.vocab = data["vocab"]
        # Ensure merges are lists of tuples, not lists of lists (JSON quirk)
        tokenizer.merges = [(tuple(pair), merged_str) for pair, merged_str in data["merges"]]

        # Rebuild ids_to_tokens from the loaded vocab
        tokenizer.ids_to_tokens = {idx: token for token, idx in tokenizer.vocab.items()}

        # Validate special token IDs after loading
        if tokenizer.pad_token and tokenizer.pad_token not in tokenizer.vocab:
            print(f"Warning: PAD token '{tokenizer.pad_token}' not found in loaded vocab.")
        if tokenizer.unk_token and tokenizer.unk_token not in tokenizer.vocab:
            print(f"Warning: UNK token '{tokenizer.unk_token}' not found in loaded vocab.")

        print(f"Tokenizer loaded from {filepath}. Vocab size: {len(tokenizer.vocab)}, Merges: {len(tokenizer.merges)}")
        return tokenizer

if __name__ == '__main__':
    print("BPE Tokenizer Example")

    # Sample corpus
    corpus_example = [
        "hello world",
        "hello there",
        "this is a test",
        "another test example",
        "hello again world",
    ]

    target_vocab_size = 50 # Small for testing, includes initial chars + merges

    print(f"\nTraining BPE tokenizer with vocab_size = {target_vocab_size}...")
    bpe_tokenizer = BPETokenizer(vocab_size=target_vocab_size)
    bpe_tokenizer.train(corpus_example, verbose=True)

    print("\nLearned Vocabulary (first 20):")
    for i, (token, token_id) in enumerate(bpe_tokenizer.vocab.items()):
        if i >= 20 and len(bpe_tokenizer.vocab) > 20 :
            print(f"... and {len(bpe_tokenizer.vocab) - 20} more")
            break
        print(f"  '{token}': {token_id}")

    print("\nLearned Merges (first 10):")
    for i, merge_rule in enumerate(bpe_tokenizer.merges):
        if i >= 10 and len(bpe_tokenizer.merges) > 10:
            print(f"... and {len(bpe_tokenizer.merges) - 10} more")
            break
        print(f"  {merge_rule[0]} -> {merge_rule[1]}")

    test_sentence = "hello world, this is a new test."
    print(f"\nEncoding sentence: '{test_sentence}'")
    encoded_ids = bpe_tokenizer.encode(test_sentence)
    print(f"Encoded IDs: {encoded_ids}")

    decoded_sentence = bpe_tokenizer.decode(encoded_ids)
    print(f"Decoded sentence: '{decoded_sentence}'")

    # Test with unknown characters / words if UNK token is used
    test_unknown = "xyz foo bar" # Assuming 'xyz', 'foo', 'bar' are not in small vocab
    print(f"\nEncoding sentence with potential unknowns: '{test_unknown}'")
    encoded_unknown = bpe_tokenizer.encode(test_unknown)
    print(f"Encoded IDs (unknowns): {encoded_unknown}")
    decoded_unknown = bpe_tokenizer.decode(encoded_unknown)
    print(f"Decoded sentence (unknowns): '{decoded_unknown}'")
    if bpe_tokenizer.unk_token_id is not None:
        print(f"(UNK token ID is {bpe_tokenizer.unk_token_id}, token is '{bpe_tokenizer.unk_token}')")


    # Save and load test
    import os
    save_path = "bpe_tokenizer_test_vocab.json"
    print(f"\nSaving tokenizer to {save_path}...")
    bpe_tokenizer.save_vocab(save_path)

    print(f"\nLoading tokenizer from {save_path}...")
    loaded_bpe_tokenizer = BPETokenizer.load_vocab(save_path)

    test_sentence_after_load = "hello again test"
    print(f"\nEncoding with loaded tokenizer: '{test_sentence_after_load}'")
    encoded_after_load = loaded_bpe_tokenizer.encode(test_sentence_after_load)
    print(f"Encoded IDs: {encoded_after_load}")
    decoded_after_load = loaded_bpe_tokenizer.decode(encoded_after_load)
    print(f"Decoded sentence: '{decoded_after_load}'")

    # Basic check: encode with original and loaded should be same
    original_encoded = bpe_tokenizer.encode(test_sentence_after_load)
    assert original_encoded == encoded_after_load, "Encoding mismatch after load!"
    print("Successfully encoded with loaded tokenizer and matched original.")

    # Clean up test file
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"\nCleaned up {save_path}")

    print("\nBPE Tokenizer Example Finished.")
