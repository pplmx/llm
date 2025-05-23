from typing import List, Dict, Set

class SimpleCharacterTokenizer:
    """
    A simple character-level tokenizer.

    This tokenizer builds a vocabulary from a given corpus and provides methods
    to encode text into a sequence of integer tokens and decode a sequence of
    tokens back into text.
    """

    def __init__(self, corpus: List[str]):
        """
        Initializes the SimpleCharacterTokenizer.

        Args:
            corpus (List[str]): A list of strings from which to build the vocabulary.
                                The vocabulary will consist of all unique characters
                                present in the corpus.
        """
        if not isinstance(corpus, list):
            raise TypeError("Corpus must be a list of strings.")
        if not all(isinstance(s, str) for s in corpus):
            raise TypeError("All items in the corpus must be strings.")

        # Join all strings in the corpus, then find unique characters
        unique_chars: Set[str] = set("".join(corpus))
        self.chars: List[str] = sorted(list(unique_chars)) # Sort for consistent mapping

        self.stoi: Dict[str, int] = {char: i for i, char in enumerate(self.chars)}
        self.itos: Dict[int, str] = {i: char for i, char in enumerate(self.chars)}
        self.vocab_size: int = len(self.chars)

        # Add PAD token
        self.pad_char: str = "<PAD>"
        if self.pad_char not in self.stoi:
            self.pad_token_id: int = self.vocab_size
            self.stoi[self.pad_char] = self.pad_token_id
            self.itos[self.pad_token_id] = self.pad_char
            self.chars.append(self.pad_char) # Add to the list of characters
            self.vocab_size += 1
        else:
            # If PAD char was part of the corpus, use its existing ID
            self.pad_token_id: int = self.stoi[self.pad_char]


    def encode(self, text: str) -> List[int]:
        """
        Encodes a string of text into a list of integer tokens.

        Args:
            text (str): The input string to encode.

        Returns:
            List[int]: A list of integer tokens representing the input text.

        Raises:
            KeyError: If the text contains characters not present in the
                      tokenizer's vocabulary (i.e., not found in the
                      initial corpus).
        """
        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")
        
        tokens: List[int] = []
        for char in text:
            try:
                tokens.append(self.stoi[char])
            except KeyError:
                raise KeyError(
                    f"Character '{char}' not found in tokenizer vocabulary. "
                    "Only characters present in the initial corpus can be encoded."
                )
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes a list of integer tokens back into a string of text.

        Args:
            tokens (List[int]): A list of integer tokens to decode.

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

        text_chars: List[str] = []
        for token in tokens:
            try:
                text_chars.append(self.itos[token])
            except KeyError:
                raise KeyError(
                    f"Token ID '{token}' not found in tokenizer vocabulary. "
                    "Only token IDs derived from the initial corpus can be decoded."
                )
        return "".join(text_chars)

if __name__ == '__main__':
    # Example Usage
    corpus_example = ["hello world!", "你好 世界！"]
    tokenizer_example = SimpleCharacterTokenizer(corpus_example)

    print(f"Vocabulary ({tokenizer_example.vocab_size} chars): {tokenizer_example.chars}") # Print list for clarity
    print(f"String to Integer mapping (stoi): {tokenizer_example.stoi}")
    print(f"Integer to String mapping (itos): {tokenizer_example.itos}")
    print(f"PAD token: '{tokenizer_example.pad_char}', ID: {tokenizer_example.pad_token_id}")
    assert tokenizer_example.itos[tokenizer_example.pad_token_id] == tokenizer_example.pad_char
    assert tokenizer_example.pad_char in tokenizer_example.stoi
    assert tokenizer_example.pad_char in tokenizer_example.chars

    text1 = "hello!"
    encoded1 = tokenizer_example.encode(text1)
    decoded1 = tokenizer_example.decode(encoded1)
    print(f"\nOriginal text 1: '{text1}'")
    print(f"Encoded: {encoded1}")
    print(f"Decoded: '{decoded1}'")
    assert decoded1 == text1

    text2 = "你好！" # Assumes these chars are in corpus_example
    encoded2 = tokenizer_example.encode(text2)
    decoded2 = tokenizer_example.decode(encoded2)
    print(f"\nOriginal text 2: '{text2}'")
    print(f"Encoded: {encoded2}")
    print(f"Decoded: '{decoded2}'")
    assert decoded2 == text2
    
    print("\nTesting empty string encoding/decoding:")
    empty_encoded = tokenizer_example.encode("")
    empty_decoded = tokenizer_example.decode([])
    print(f"Encoded empty string: {empty_encoded}")
    print(f"Decoded empty list: '{empty_decoded}'")
    assert empty_encoded == []
    assert empty_decoded == ""

    print("\nTesting unknown character (should raise KeyError):")
    try:
        tokenizer_example.encode("abc") # 'a', 'b', 'c' are not in the example corpus
    except KeyError as e:
        print(f"Caught expected error for unknown char: {e}")

    print("\nTesting unknown token ID (should raise KeyError):")
    unknown_id = tokenizer_example.vocab_size + 10 # An ID guaranteed to be out of vocab
    try:
        tokenizer_example.decode([unknown_id]) 
    except KeyError as e:
        print(f"Caught expected error for unknown token: {e}")

    print("\nTesting decoding of PAD token:")
    decoded_pad = tokenizer_example.decode([tokenizer_example.pad_token_id])
    print(f"Decoded PAD token ID [{tokenizer_example.pad_token_id}]: '{decoded_pad}'")
    assert decoded_pad == tokenizer_example.pad_char
    
    print("\nAll basic __main__ tests passed.")
