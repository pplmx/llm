from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from llm.tokenization.tokenizer import BaseTokenizer


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
        tokenizer: BaseTokenizer,
        max_seq_len: int,
        overlap: int = 0,
        padding_value: int | None = None,  # Allow None to use tokenizer's pad_id
    ):
        """
        Initializes the TextDataset.

        Args:
            file_path (str): Path to the text file.
            tokenizer (BaseTokenizer): A tokenizer instance satisfying the BaseTokenizer protocol.
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
        else:
            self.padding_value = padding_value

        # Read and tokenize the entire text file
        try:
            with Path(self.file_path).open(encoding="utf-8") as f:
                text_content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        except PermissionError:
            raise PermissionError(f"Permission denied reading file: {self.file_path}")
        except OSError as e:
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
    """Build a simple DataLoader for a TextDataset (scripts / legacy tests)."""
    return build_text_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_text_dataloader(
    dataset: TextDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Build a PyTorch DataLoader for a TextDataset.

    Prefer ``TokenizedMapDataModule`` for training; this helper is for scripts.
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
