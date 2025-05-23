import pytest
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import tempfile
import os

# Adjust path to import from src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from llm.data.loader import TextDataset, create_dataloader
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer

# Constants for tests
MAX_SEQ_LEN = 10
DEFAULT_OVERLAP = 2
SAMPLE_CORPUS = ["abc", "def", "ghi", "<PAD>"] # Ensure PAD is part of corpus for consistent tokenizer

@pytest.fixture(scope="session")
def dummy_tokenizer():
    """A session-scoped tokenizer based on a small fixed corpus."""
    # SimpleCharacterTokenizer now adds <PAD> if not present.
    # If corpus is ["a", "b"], vocab will be ['a', 'b', '<PAD>']
    # pad_token_id will be 2.
    # If corpus is ["a", "b", "<PAD>"], vocab will be ['<PAD>', 'a', 'b'] (sorted) or similar,
    # and pad_token_id will be its index.
    # The current SimpleCharacterTokenizer appends <PAD> if not in corpus chars.
    # So, for corpus ["a","b"], chars=['a','b'], then pad_char="<PAD>", pad_token_id=2, chars=['a','b','<PAD>']
    return SimpleCharacterTokenizer(SAMPLE_CORPUS)

@pytest.fixture
def dummy_text_file(request):
    """Creates a temporary text file with specified content."""
    content = getattr(request, "param", "default testing content. " * 5) # Default content
    
    # Using NamedTemporaryFile to ensure it's cleaned up, but need to close it
    # before TextDataset can open it by path, especially on Windows.
    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8")
    temp_file.write(content)
    temp_file_path = temp_file.name
    temp_file.close() # Close the file so it can be opened by TextDataset

    yield Path(temp_file_path) # Provide the path to the test

    os.remove(temp_file_path) # Manual cleanup after the test using this fixture

class TestTextDatasetInitialization:
    @pytest.mark.parametrize("dummy_text_file", ["short example text."], indirect=True)
    def test_dataset_creation(self, dummy_text_file, dummy_tokenizer):
        dataset = TextDataset(
            file_path=str(dummy_text_file),
            tokenizer=dummy_tokenizer,
            max_seq_len=MAX_SEQ_LEN
        )
        assert len(dataset) > 0 # Should have at least one sequence
        assert dataset.padding_value == dummy_tokenizer.pad_token_id

    @pytest.mark.parametrize("dummy_text_file", ["a b c d e f g h i j k l m n o p q r s t"], indirect=True) # 20 tokens if space is token
    def test_chunking_no_overlap(self, dummy_text_file, dummy_tokenizer):
        # Assuming tokenizer splits by space for simplicity of token counting here
        # SimpleCharacterTokenizer will tokenize each char. Content: "a b c..." (39 chars)
        # Vocab: ' ', 'a', 'b', ..., 't', '<PAD>'
        # Encoded length will be 39.
        # max_seq_len = 10, overlap = 0. Expected chunks: 39 // 10 + (1 if 39 % 10 else 0) = 3 + 1 = 4
        dataset = TextDataset(
            file_path=str(dummy_text_file),
            tokenizer=dummy_tokenizer,
            max_seq_len=10,
            overlap=0
        )
        # text = "a b c d e f g h i j k l m n o p q r s t" (length 39 chars)
        # tokens = tokenizer.encode(text) -> len(tokens) = 39
        # Chunks (max_len=10, step=10): [0:10], [10:20], [20:30], [30:39] -> 4 sequences
        assert len(dataset) == 4 
        # First sequence should be first 10 tokens of encoded text
        text_content = dummy_text_file.read_text(encoding="utf-8")
        all_tokens = dummy_tokenizer.encode(text_content)
        expected_first_seq = all_tokens[:10]
        actual_first_seq_raw = dataset.sequences[0] # Before padding in __getitem__
        assert actual_first_seq_raw == expected_first_seq

    @pytest.mark.parametrize("dummy_text_file", ["a b c d e f g h i j k l m n o p q r s t"], indirect=True) # 39 chars
    def test_chunking_with_overlap(self, dummy_text_file, dummy_tokenizer):
        max_len = 10
        overlap = 2
        step = max_len - overlap # 8
        dataset = TextDataset(
            file_path=str(dummy_text_file),
            tokenizer=dummy_tokenizer,
            max_seq_len=max_len,
            overlap=overlap
        )
        text_content = dummy_text_file.read_text(encoding="utf-8")
        all_tokens = dummy_tokenizer.encode(text_content) # len 39
        # Chunks (max_len=10, step=8):
        # 0: [0:10]
        # 1: [8:18]
        # 2: [16:26]
        # 3: [24:34]
        # 4: [32:39] (shorter raw sequence)
        # Expected number of sequences: ceil((len(all_tokens) - max_len) / step) + 1 if len > max_len
        # or more simply: (len(all_tokens) - overlap) // step, then adjust for first full one.
        # (39 - 10) / 8 + 1 = 29 / 8 + 1 = 3 + 1 = 4. No, this is not general.
        # (39 - overlap) // step = (39-2)//8 = 37 // 8 = 4.  This is num_steps after first. So 5 sequences.
        # More general: (L - M) // (M - O) + 1, then handle remainder.
        # Or simply count:
        count = 0
        for i in range(0, len(all_tokens), step):
            if i + max_len <= len(all_tokens) or (i < len(all_tokens) and i + max_len > len(all_tokens)): # last chunk condition
                 if len(all_tokens[i:i+max_len]) > 0 or len(all_tokens[i:]) > 0 and i < len(all_tokens): # ensure last chunk has content
                    count +=1
            # A simpler way: (len(all_tokens) + step -1 - max_len % step ) // step if we only take full steps
            # For this chunking logic: number of sequences is len(range(0, L, S))
        
        expected_num_sequences = (len(all_tokens) - 1) // step + 1 if len(all_tokens) > 0 else 0
        # Example: L=39, M=10, O=2, S=8. (39-1)//8 + 1 = 38//8 + 1 = 4+1=5. Correct.
        # Example: L=10, M=10, O=0, S=10. (10-1)//10 + 1 = 0+1=1. Correct.
        # Example: L=5, M=10, O=0, S=10. (5-1)//10+1 = 0+1=1. Correct.
        # Example: L=0, M=10, O=0, S=10. -> 0. Correct.

        assert len(dataset) == expected_num_sequences
        assert dataset.sequences[0] == all_tokens[0:max_len]
        assert dataset.sequences[1] == all_tokens[step : step + max_len]


    @pytest.mark.parametrize("dummy_text_file", [""], indirect=True) # Empty file
    def test_empty_file(self, dummy_text_file, dummy_tokenizer):
        dataset = TextDataset(str(dummy_text_file), dummy_tokenizer, MAX_SEQ_LEN)
        assert len(dataset) == 0

    @pytest.mark.parametrize("dummy_text_file", ["short"], indirect=True) # Text shorter than max_seq_len
    def test_text_shorter_than_max_seq_len(self, dummy_text_file, dummy_tokenizer):
        dataset = TextDataset(str(dummy_text_file), dummy_tokenizer, MAX_SEQ_LEN)
        assert len(dataset) == 1
        item = dataset[0] # Test __getitem__ padding
        assert item["input_ids"].shape == (MAX_SEQ_LEN,)
        
        text_content = dummy_text_file.read_text(encoding="utf-8")
        expected_raw_tokens = dummy_tokenizer.encode(text_content)
        num_actual_tokens = len(expected_raw_tokens)
        
        assert torch.equal(item["input_ids"][:num_actual_tokens], torch.LongTensor(expected_raw_tokens))
        assert torch.all(item["input_ids"][num_actual_tokens:] == dummy_tokenizer.pad_token_id)


    def test_invalid_params(self, dummy_text_file, dummy_tokenizer):
        with pytest.raises(ValueError, match="overlap must be less than max_seq_len"):
            TextDataset(str(dummy_text_file), dummy_tokenizer, 5, overlap=5)
        with pytest.raises(FileNotFoundError):
            TextDataset("non_existent_file.txt", dummy_tokenizer, MAX_SEQ_LEN)

class TestTextDatasetGetItem:
    @pytest.mark.parametrize("dummy_text_file", ["hello there general kenobi"], indirect=True)
    def test_getitem_output(self, dummy_text_file, dummy_tokenizer):
        dataset = TextDataset(str(dummy_text_file), dummy_tokenizer, MAX_SEQ_LEN)
        item = dataset[0]
        assert "input_ids" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert item["input_ids"].dtype == torch.long
        assert item["input_ids"].shape == (MAX_SEQ_LEN,)
        assert torch.equal(item["input_ids"], item["labels"])

    @pytest.mark.parametrize("dummy_text_file", ["this is the last sequence, and it is short"], indirect=True)
    def test_getitem_padding_last_sequence(self, dummy_text_file, dummy_tokenizer):
        # Choose max_seq_len and overlap such that the last sequence is shorter.
        # Text length: 46. Tokens: 46.
        # max_len=20, overlap=5, step=15
        # seq1: [0:20]
        # seq2: [15:35]
        # seq3: [30:46] (length 16, needs padding)
        max_len = 20
        dataset = TextDataset(str(dummy_text_file), dummy_tokenizer, max_len, overlap=5)
        assert len(dataset) == 3
        
        last_item = dataset[2]
        assert last_item["input_ids"].shape == (max_len,)
        
        text_content = dummy_text_file.read_text(encoding="utf-8")
        all_tokens = dummy_tokenizer.encode(text_content)
        expected_last_raw_seq = all_tokens[30:] # Tokens from index 30 to end
        num_actual_tokens = len(expected_last_raw_seq)
        assert num_actual_tokens < max_len # Ensure it was indeed shorter

        assert torch.equal(last_item["input_ids"][:num_actual_tokens], torch.LongTensor(expected_last_raw_seq))
        assert torch.all(last_item["input_ids"][num_actual_tokens:] == dummy_tokenizer.pad_token_id)

    @pytest.mark.parametrize("dummy_text_file", ["some data"], indirect=True)
    def test_getitem_out_of_bounds(self, dummy_text_file, dummy_tokenizer):
        dataset = TextDataset(str(dummy_text_file), dummy_tokenizer, MAX_SEQ_LEN)
        with pytest.raises(IndexError):
            _ = dataset[len(dataset)] # Access one index beyond the end

class TestCreateDataLoader:
    @pytest.mark.parametrize("dummy_text_file", ["batch data " * 20], indirect=True) # Enough for multiple batches
    def test_dataloader_creation_and_iteration(self, dummy_text_file, dummy_tokenizer):
        dataset = TextDataset(str(dummy_text_file), dummy_tokenizer, MAX_SEQ_LEN, overlap=DEFAULT_OVERLAP)
        batch_size = 2
        dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)

        assert isinstance(dataloader, DataLoader)
        
        batch = next(iter(dataloader)) # Get one batch
        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape == (batch_size, MAX_SEQ_LEN)
        assert batch["labels"].shape == (batch_size, MAX_SEQ_LEN)
        assert batch["input_ids"].dtype == torch.long
        assert batch["labels"].dtype == torch.long

    @pytest.mark.parametrize("dummy_text_file", ["single item dataset"], indirect=True)
    def test_dataloader_last_batch_handling(self, dummy_text_file, dummy_tokenizer):
        # If dataset size is not a multiple of batch_size.
        # This text creates 1 sequence of length 19. Padded to MAX_SEQ_LEN=10 (oops, this is too short)
        # Let's make MAX_SEQ_LEN = 20 for this test.
        current_max_seq_len = 20
        dataset = TextDataset(str(dummy_text_file), dummy_tokenizer, current_max_seq_len, overlap=0)
        # Content "single item dataset" is 19 chars. So 1 sequence.
        assert len(dataset) == 1
        
        dataloader = create_dataloader(dataset, batch_size=4, shuffle=False) # Batch size > dataset length
        batch = next(iter(dataloader))
        assert batch["input_ids"].shape == (1, current_max_seq_len) # Batch size should be 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

```
