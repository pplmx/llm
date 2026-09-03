import json

import pytest
import torch

from llm.data.datasets.sft import SFTDataset
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


def test_sft_dataset(tmp_path):
    # Setup dummy data
    data = [
        {"instruction": "Hi", "input": "", "output": "Hello"},
        {"instruction": "Add", "input": "1+1", "output": "2"},
    ]
    file_path = tmp_path / "train.jsonl"
    with file_path.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    # Setup tokenizer
    # Use printable characters to cover all template static text
    from string import printable

    tokenizer = SimpleCharacterTokenizer(corpus=[printable])
    # Add special tokens just in case, though SimpleTokenizer handles PAD automatically if needed.
    # But wait, SimpleCharacterTokenizer builds vocab from corpus chars.
    # PAD is usually added if not there?
    # Let's ensure it works. SimpleCharacterTokenizer is robust enough.

    # Create dataset
    max_len = 200  # long enough
    dataset = SFTDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_seq_len=max_len,
        padding_value=0,  # assumed pad
        ignore_index=-100,
    )

    # Test length
    assert len(dataset) == 2

    # Test first item
    item = dataset[0]
    input_ids = item["input_ids"]
    labels = item["labels"]
    attn_mask = item["attention_mask"]

    assert input_ids.shape == (max_len,)
    assert labels.shape == (max_len,)
    assert attn_mask.shape == (max_len,)

    # Verify masking
    # Reconstruct prompt to find prompt length
    prompt, response = dataset.alpaca_template(data[0])
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response)

    expected_len = len(prompt_ids) + len(response_ids)

    # Check inputs match concatenation
    # Note: excluding padding
    valid_len = sum(attn_mask).item()
    assert valid_len == expected_len

    assert torch.equal(input_ids[: len(prompt_ids)], torch.tensor(prompt_ids))
    assert torch.equal(input_ids[len(prompt_ids) : valid_len], torch.tensor(response_ids))

    # Check labels
    # Prompt part should be -100
    assert torch.all(labels[: len(prompt_ids)] == -100)
    # Response part should match input_ids
    assert torch.equal(labels[len(prompt_ids) : valid_len], input_ids[len(prompt_ids) : valid_len])
    # Padding part should be -100
    assert torch.all(labels[valid_len:] == -100)


def test_sft_dataset_truncation_keeps_response_tail(tmp_path):
    """Truncation drops the PROMPT PREFIX, never the supervised response tail.

    Regression (RIL ISS-332): the old ``input_ids[:max_seq_len]`` kept the
    whole prompt and chopped the response end — the only supervised signal,
    since prompt tokens are label-masked. With a pair whose prompt alone
    exceeds the window, the KEPT tokens must be the response end with live
    labels, not an all-masked row.
    """
    data = [{"instruction": "A" * 100, "input": "", "output": "B" * 100}]
    file_path = tmp_path / "trunc.jsonl"
    with file_path.open("w") as f:
        f.write(json.dumps(data[0]) + "\n")

    from string import printable

    tokenizer = SimpleCharacterTokenizer([printable])
    dataset = SFTDataset(file_path, tokenizer, max_seq_len=10)  # very short

    item = dataset[0]
    assert item["input_ids"].shape == (10,)
    assert item["labels"].shape == (10,)

    # The kept 10 tokens are the END of prompt+response (the "B" response tail).
    prompt, response = dataset.alpaca_template(data[0])
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response)
    combined = prompt_ids + response_ids
    expected_input = combined[-10:]
    expected_labels = ([-100] * len(prompt_ids) + response_ids)[-10:]

    assert item["input_ids"].tolist() == expected_input
    assert item["labels"].tolist() == expected_labels
    # The supervised payload survived — not an all-masked row.
    assert not torch.all(item["labels"] == -100)
    # Nothing was padded (truncation fills the window), so the mask is all 1s.
    assert item["attention_mask"].tolist() == [1] * 10


def test_sft_dataset_rejects_nonpositive_max_seq_len(tmp_path):
    """RIL ISS-199: a non-positive ``max_seq_len`` fails fast instead of
    silently truncating ids and misaligning attention_mask against
    input_ids."""
    from string import printable

    from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer

    fp = tmp_path / "x.jsonl"
    fp.write_text('{"instruction":"Hi","input":"","output":"Hello"}\n', encoding="utf-8")
    tok = SimpleCharacterTokenizer([printable])
    with pytest.raises(ValueError, match="max_seq_len"):
        SFTDataset(file_path=fp, tokenizer=tok, max_seq_len=0)
    with pytest.raises(ValueError, match="max_seq_len"):
        SFTDataset(file_path=fp, tokenizer=tok, max_seq_len=-1)
