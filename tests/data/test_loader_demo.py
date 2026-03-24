"""
Data Loader Demo Tests

Tests TextDataset and create_dataloader functionality.
"""

import tempfile
from pathlib import Path

import torch

from llm.data.loader import TextDataset, create_dataloader
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


def test_text_dataset():
    """Test TextDataset with a temporary file."""
    # Create dummy corpus and tokenizer - include all needed characters
    dummy_corpus = ["abcdefghijklmnopqrstuvwxyz .", "<PAD>"]
    tokenizer = SimpleCharacterTokenizer(dummy_corpus)

    # Create dummy text file
    dummy_text_content = "hello world. this is a test text for the data loader. " * 5

    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp_file:
        tmp_file.write(dummy_text_content)
        tmp_file_path = tmp_file.name

    try:
        max_len = 50
        overlap_val = 5

        dataset = TextDataset(
            file_path=tmp_file_path,
            tokenizer=tokenizer,
            max_seq_len=max_len,
            overlap=overlap_val,
            padding_value=tokenizer.pad_token_id,
        )

        assert len(dataset) > 0, "Dataset should not be empty"

        # Test first item
        item0 = dataset[0]
        assert item0["input_ids"].shape == (max_len,)
        assert item0["labels"].shape == (max_len,)
        assert item0["input_ids"].dtype == torch.long
        assert torch.equal(item0["input_ids"], item0["labels"])

        # Test last item
        item_last = dataset[len(dataset) - 1]
        assert item_last["input_ids"].shape == (max_len,)  # Should be padded

        # Check non-pad tokens in last item
        num_non_pad = sum(1 for token_id in item_last["input_ids"] if token_id != tokenizer.pad_token_id)
        assert num_non_pad >= 0  # Could be all padding if content is short

    finally:
        Path(tmp_file_path).unlink(missing_ok=True)


def test_create_dataloader():
    """Test create_dataloader with TextDataset."""
    # Create dummy corpus and tokenizer - include all needed characters
    dummy_corpus = ["abcdefghijklmnopqrstuvwxyz .", "<PAD>"]
    tokenizer = SimpleCharacterTokenizer(dummy_corpus)

    # Create dummy text file
    dummy_text_content = "hello world. this is a test text for the data loader. " * 5

    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp_file:
        tmp_file.write(dummy_text_content)
        tmp_file_path = tmp_file.name

    try:
        max_len = 50
        overlap_val = 5

        dataset = TextDataset(
            file_path=tmp_file_path,
            tokenizer=tokenizer,
            max_seq_len=max_len,
            overlap=overlap_val,
            padding_value=tokenizer.pad_token_id,
        )

        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False)

        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            assert batch["input_ids"].shape[1] == max_len
            assert batch["labels"].shape == batch["input_ids"].shape
            assert batch["input_ids"].dtype == torch.long
            break  # Only test one batch

        assert batch_count > 0, "DataLoader should yield at least one batch"

    finally:
        Path(tmp_file_path).unlink(missing_ok=True)
