import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


class TextDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing text data for language modeling.

    The dataset reads a text file, tokenizes it, and creates overlapping or
    non-overlapping sequences of a fixed maximum length. Shorter sequences
    (typically the last one) are padded.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: Any,  # Should ideally be a more specific TokenizerProtocol
        max_seq_len: int,
        overlap: int = 0,
        padding_value: int | None = None,  # Allow None to use tokenizer's pad_id
    ):
        """
        Initializes the TextDataset.

        Args:
            file_path (str): Path to the text file.
            tokenizer (Any): A tokenizer instance with `encode` and `pad_token_id` attributes.
            max_seq_len (int): The maximum length for each sequence.
            overlap (int, default=0): The number of tokens to overlap between consecutive sequences.
                                      Must be less than `max_seq_len`.
            padding_value (int, optional): Value to use for padding shorter sequences.
                                           If None, defaults to `tokenizer.pad_token_id`.
                                           If tokenizer has no `pad_token_id`, defaults to 0.
        """
        if not isinstance(file_path, str | Path):
            raise TypeError("file_path must be a string or Path object.")

        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        if not hasattr(tokenizer, "encode") or not callable(tokenizer.encode):
            raise ValueError("Tokenizer must have an 'encode' method.")

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if not isinstance(max_seq_len, int) or max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer.")

        if not isinstance(overlap, int) or overlap < 0:
            raise ValueError("overlap must be a non-negative integer.")
        if overlap >= max_seq_len:
            raise ValueError("overlap must be less than max_seq_len.")

        self.overlap = overlap

        if padding_value is None:
            if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
                self.padding_value = self.tokenizer.pad_token_id
            else:
                # Fallback if tokenizer doesn't specify a pad_token_id
                # (though SimpleCharacterTokenizer is now expected to have one)
                self.padding_value = 0
                # print(f"Warning: padding_value not specified and tokenizer has no pad_token_id. Defaulting to 0.", file=sys.stderr)
                # Avoiding print to stderr for cleaner test output, assuming this case is handled or tested elsewhere.
        else:
            self.padding_value = padding_value

        # Read and tokenize the entire text file
        try:
            with open(self.file_path, encoding="utf-8") as f:
                text_content = f.read()
        except Exception as e:
            raise OSError(f"Error reading file {self.file_path}: {e}")

        if not text_content:  # Handle empty file
            self.sequences: list[list[int]] = []
            return

        all_token_ids: list[int] = self.tokenizer.encode(text_content)

        # Chunk token_ids into sequences
        self.sequences = []
        step = self.max_seq_len - self.overlap
        if step <= 0:  # Should be caught by overlap < max_seq_len check, but as a safeguard
            raise ValueError("Step size (max_seq_len - overlap) must be positive.")

        for i in range(0, len(all_token_ids), step):
            chunk = all_token_ids[i : i + self.max_seq_len]
            # We don't pad here yet; padding happens in __getitem__ to ensure all items
            # from __getitem__ have the same length. Chunks here can be shorter if at end.
            if not chunk:  # Should not happen if all_token_ids is not empty
                continue

            # Only add chunks that have some content. If a chunk would be entirely padding
            # due to being far past the end of all_token_ids, it might be skipped.
            # However, range(0, len, step) ensures i is always a valid start.
            # If len(all_token_ids) is small, e.g., less than max_seq_len, one chunk is added.
            if len(chunk) > 0:  # Ensure we don't add empty lists if somehow a step lands weirdly
                self.sequences.append(chunk)

        # Handle case where the last sequence might be shorter than max_seq_len and step
        # The loop structure already correctly creates the last chunk, which might be shorter.
        # For example, if len(all_token_ids) = 15, max_seq_len=10, overlap=0 (step=10)
        # i=0, chunk = tokens[0:10] -> self.sequences.append(tokens[0:10])
        # i=10, chunk = tokens[10:15] -> self.sequences.append(tokens[10:15])
        # This is correct. Padding is handled in __getitem__.

    def __len__(self) -> int:
        """Returns the number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Retrieves a single data item (input_ids and labels) at the given index.

        The sequence is padded to `max_seq_len` if it's shorter.
        Labels are created as a clone of the input_ids.

        Args:
            idx (int): Index of the sequence to retrieve.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing:
                - "input_ids": Padded token IDs (torch.LongTensor).
                - "labels": Cloned padded token IDs (torch.LongTensor).
        """
        if not 0 <= idx < len(self.sequences):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.sequences)}")

        token_ids: list[int] = self.sequences[idx]

        # Pad the sequence if it's shorter than max_seq_len
        num_padding_tokens = self.max_seq_len - len(token_ids)
        if num_padding_tokens > 0:
            padded_token_ids = token_ids + [self.padding_value] * num_padding_tokens
        else:
            # If somehow a sequence was longer (shouldn't happen with current chunking), truncate.
            # Or, if exactly max_seq_len, no change.
            padded_token_ids = token_ids[: self.max_seq_len]

        input_ids_tensor = torch.LongTensor(padded_token_ids)
        labels_tensor = input_ids_tensor.clone()  # Labels are same as input for typical LM

        return {"input_ids": input_ids_tensor, "labels": labels_tensor}


def create_dataloader(
    dataset: TextDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Creates a PyTorch DataLoader for a TextDataset.

    Args:
        dataset (TextDataset): The dataset to load.
        batch_size (int): Number of samples per batch.
        shuffle (bool, default=True): Whether to shuffle the data at every epoch.
        num_workers (int, default=0): Number of subprocesses to use for data loading.
        pin_memory (bool, default=False): If True, copies Tensors into CUDA pinned memory
                                          before returning them (if using CUDA).

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """
    if not isinstance(dataset, TextDataset):
        raise TypeError("dataset must be an instance of TextDataset.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    # Since TextDataset.__getitem__ ensures all tensors are padded to max_seq_len,
    # the default collate_fn should work fine.
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # collate_fn=None, # Use default collate_fn
    )


if __name__ == "__main__":
    # Example Usage (requires a dummy tokenizer and a text file)
    # os and sys are imported at the top of the file.
    from tempfile import NamedTemporaryFile

    # Create a dummy tokenizer (SimpleCharacterTokenizer is now expected to have PAD)
    dummy_corpus = ["abcdefghijklmnopqrstuvwxyz ", "<PAD>"]  # Ensure PAD is in corpus for this test
    # Note: SimpleCharacterTokenizer adds <PAD> if not present.
    # If corpus already has <PAD>, it will use its ID.
    # If corpus does not have <PAD>, it adds it.

    # We need to ensure SimpleCharacterTokenizer is available.
    # Let's assume it is for the __main__ part.
    tmp_file_path = None  # Initialize to ensure it's defined in finally block
    try:
        tokenizer = SimpleCharacterTokenizer(dummy_corpus)
        print(f"Tokenizer created. Vocab size: {tokenizer.vocab_size}, PAD ID: {tokenizer.pad_token_id}")

        # Create a dummy text file
        dummy_text_content = "hello world. this is a test text for the data loader. " * 5
        # Approx 50 chars * 5 = 250 chars.
        # If max_seq_len = 50, overlap = 5 (step=45)
        # Chunks: 0-50, 45-95, 90-140, 135-185, 180-230, 225-250 (last one shorter)

        with NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp_file:
            tmp_file.write(dummy_text_content)
            tmp_file_path = tmp_file.name

        print(f"\nCreated dummy text file: {tmp_file_path} with ~{len(dummy_text_content)} chars.")

        # Test TextDataset
        print("\nTesting TextDataset...")
        max_len = 50
        overlap_val = 5
        dataset = TextDataset(
            file_path=tmp_file_path,
            tokenizer=tokenizer,
            max_seq_len=max_len,
            overlap=overlap_val,
            padding_value=tokenizer.pad_token_id,  # Explicitly pass pad_token_id
        )
        print(f"  Dataset length: {len(dataset)}")
        assert len(dataset) > 0, "Dataset should not be empty with this content."

        print("  Getting first item (idx=0) from dataset:")
        item0 = dataset[0]
        print(f"    input_ids shape: {item0['input_ids'].shape}, dtype: {item0['input_ids'].dtype}")
        print(f"    labels shape: {item0['labels'].shape}, dtype: {item0['labels'].dtype}")
        assert item0["input_ids"].shape == (max_len,)
        assert item0["labels"].shape == (max_len,)
        assert item0["input_ids"].dtype == torch.long
        assert torch.equal(item0["input_ids"], item0["labels"])

        print(f"  Getting last item (idx={len(dataset) - 1}) from dataset:")
        item_last = dataset[len(dataset) - 1]
        print(f"    input_ids shape: {item_last['input_ids'].shape}")
        # Check if padding was applied by seeing if last part is pad_token_id
        # The actual content of the last chunk of all_token_ids:
        all_tokens = tokenizer.encode(dummy_text_content)
        # Calculate expected length of actual content in the last chunk
        # This is more reliable than calculating from len(all_tokens) % step.
        num_non_pad = 0
        for token_id_val in item_last["input_ids"]:  # Renamed to avoid conflict with outer scope
            if token_id_val != tokenizer.pad_token_id:
                num_non_pad += 1
        print(f"    Number of non-pad tokens in last item's input_ids: {num_non_pad}")
        assert item_last["input_ids"].shape == (max_len,)  # Should be padded

        # Test DataLoader
        print("\nTesting DataLoader...")
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False)
        print("  DataLoader created. Iterating through one batch...")

        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            print(f"    Batch {batch_count}:")
            print(f"      input_ids shape: {batch['input_ids'].shape}, dtype: {batch['input_ids'].dtype}")
            print(f"      labels shape: {batch['labels'].shape}, dtype: {batch['labels'].dtype}")
            assert (
                batch["input_ids"].shape == (2, max_len) or batch["input_ids"].shape[1] == max_len
            )  # Handles last partial batch
            assert batch["labels"].shape == batch["input_ids"].shape
            assert batch["input_ids"].dtype == torch.long
            break  # Only test one batch

        assert batch_count > 0, "DataLoader did not yield any batches."

        print("\nAll basic __main__ tests passed.")

    except ImportError:
        print("SimpleCharacterTokenizer not found, skipping __main__ example for data loader.", file=sys.stderr)
    except Exception as e:
        print(f"Error in __main__ example: {e}", file=sys.stderr)
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            print(f"Removed dummy text file: {tmp_file_path}")
