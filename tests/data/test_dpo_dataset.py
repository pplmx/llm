import json
from string import printable

import pytest
import torch

from llm.data.datasets.dpo import DPODataset
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


@pytest.fixture
def tokenizer():
    return SimpleCharacterTokenizer([printable])


@pytest.fixture
def data_file(tmp_path):
    data = [{"prompt": "Q:", "chosen": "Good", "rejected": "Bad"}]
    file_path = tmp_path / "dpo_data.jsonl"
    with file_path.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return file_path


def test_dpo_dataset(data_file, tokenizer):
    dataset = DPODataset(file_path=data_file, tokenizer=tokenizer, max_seq_len=20)

    assert len(dataset) == 1
    item = dataset[0]

    assert "chosen_input_ids" in item
    assert "rejected_input_ids" in item
    assert "chosen_labels" in item

    # Verify content
    prompt_ids = tokenizer.encode("Q:")
    chosen_ids = tokenizer.encode("Good")
    expected_chosen_len = len(prompt_ids) + len(chosen_ids)

    assert item["chosen_input_ids"][:expected_chosen_len].tolist() == prompt_ids + chosen_ids
    # Check Masking (Prompt masked)
    assert torch.all(item["chosen_labels"][: len(prompt_ids)] == -100)
    assert item["chosen_labels"][len(prompt_ids) : expected_chosen_len].tolist() == chosen_ids

    # Check Rejected
    rejected_ids = tokenizer.encode("Bad")
    expected_rejected_len = len(prompt_ids) + len(rejected_ids)
    assert item["rejected_input_ids"][:expected_rejected_len].tolist() == prompt_ids + rejected_ids
    assert torch.all(item["rejected_labels"][: len(prompt_ids)] == -100)


def test_dpo_dataset_file_not_found(tokenizer):
    """FileNotFoundError raised when file does not exist."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        DPODataset(file_path="/nonexistent/path.jsonl", tokenizer=tokenizer)


def test_dpo_dataset_empty_lines_skipped(tmp_path, tokenizer):
    """Empty lines in the JSONL file are skipped."""
    file_path = tmp_path / "empty_lines.jsonl"
    with file_path.open("w") as f:
        f.write('{"prompt": "Q1", "chosen": "A1", "rejected": "R1"}\n')
        f.write("\n")  # Empty line
        f.write("\n")  # Another empty line
        f.write('{"prompt": "Q2", "chosen": "A2", "rejected": "R2"}\n')

    dataset = DPODataset(file_path=file_path, tokenizer=tokenizer, max_seq_len=20)
    assert len(dataset) == 2


def test_dpo_dataset_missing_keys_skipped(tmp_path, tokenizer):
    """Entries missing required keys (prompt, chosen, rejected) are skipped."""
    file_path = tmp_path / "missing_keys.jsonl"
    with file_path.open("w") as f:
        f.write('{"prompt": "Q1", "chosen": "A1", "rejected": "R1"}\n')
        f.write('{"prompt": "Q2", "chosen": "A2"}\n')  # Missing "rejected"
        f.write('{"prompt": "Q3", "rejected": "R3"}\n')  # Missing "chosen"
        f.write('{"chosen": "A4", "rejected": "R4"}\n')  # Missing "prompt"

    dataset = DPODataset(file_path=file_path, tokenizer=tokenizer, max_seq_len=20)
    # Only the first entry is valid
    assert len(dataset) == 1


def test_dpo_dataset_json_decode_error(tmp_path, tokenizer):
    """Invalid JSON lines raise ValueError."""
    file_path = tmp_path / "bad_json.jsonl"
    with file_path.open("w") as f:
        f.write("not valid json\n")

    with pytest.raises(ValueError, match="Invalid JSON"):
        DPODataset(file_path=file_path, tokenizer=tokenizer, max_seq_len=20)


def test_dpo_dataset_truncation(tmp_path, tokenizer):
    """Sequences exceeding max_seq_len are truncated."""
    file_path = tmp_path / "truncate.jsonl"
    with file_path.open("w") as f:
        f.write(
            '{"prompt": "P:", "chosen": "This is a very long chosen response that exceeds the max length", "rejected": "Short"}'
        )

    # max_seq_len=5 forces truncation for the long chosen text
    dataset = DPODataset(file_path=file_path, tokenizer=tokenizer, max_seq_len=5)
    item = dataset[0]

    # Each item has chosen AND rejected data
    assert item["chosen_input_ids"].shape[0] == 5
    assert item["rejected_input_ids"].shape[0] == 5
    # Labels should also be truncated
    assert item["chosen_labels"].shape[0] == 5
    assert item["rejected_labels"].shape[0] == 5


def test_dpo_dataset_padding_small_sequence(tmp_path, tokenizer):
    """Short sequences are padded to max_seq_len."""
    file_path = tmp_path / "padding.jsonl"
    with file_path.open("w") as f:
        f.write('{"prompt": "X", "chosen": "Y", "rejected": "Z"}')

    # max_seq_len=10 > actual length → padding applied
    dataset = DPODataset(file_path=file_path, tokenizer=tokenizer, max_seq_len=10)
    item = dataset[0]

    assert item["chosen_input_ids"].shape[0] == 10
    assert item["rejected_input_ids"].shape[0] == 10
    # Padding tokens should be at the end
    prompt_ids = tokenizer.encode("X")
    chosen_ids = tokenizer.encode("Y")
    seq_len = len(prompt_ids) + len(chosen_ids)
    # The padding_value is 0, so last tokens should be 0
    assert item["chosen_input_ids"][seq_len:].tolist() == [0] * (10 - seq_len)
    # Padding labels should be -100
    assert item["chosen_labels"][seq_len:].tolist() == [-100] * (10 - seq_len)


def test_dpo_dataset_overlong_prompt_dropped(tmp_path, tokenizer):
    """A prompt that alone reaches max_seq_len truncates the completion
    ENTIRELY (all-(-100) labels = an empty preference signal that would
    silently contribute a constant log(2) to the DPO loss). The row is dropped
    at load instead of trained on (deep-dive finding)."""
    long_prompt = "P" * 30  # > max_seq_len=20 tokens
    file_path = tmp_path / "overlong.jsonl"
    with file_path.open("w") as f:
        f.write('{"prompt": "Q1", "chosen": "A1", "rejected": "R1"}\n')
        f.write(json.dumps({"prompt": long_prompt, "chosen": "Good", "rejected": "Bad"}) + "\n")
        f.write('{"prompt": "Q2", "chosen": "A2", "rejected": "R2"}\n')

    dataset = DPODataset(file_path=file_path, tokenizer=tokenizer, max_seq_len=20)
    assert len(dataset) == 2, "the over-long-prompt row must be dropped"


def test_dpo_dataset_directory_as_file(tmp_path, tokenizer):
    """OSError raised when file path is a directory."""
    with pytest.raises(OSError, match="Error reading DPO file"):
        DPODataset(file_path=tmp_path, tokenizer=tokenizer, max_seq_len=20)


def test_dpo_dataset_rejects_nonpositive_max_seq_len(tmp_path, tokenizer):
    """RIL ISS-199: a non-positive ``max_seq_len`` fails fast instead of
    silently truncating ids and misaligning attention_mask against
    input_ids."""
    fp = tmp_path / "x.jsonl"
    fp.write_text('{"prompt":"Q:","chosen":"Good","rejected":"Bad"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="max_seq_len"):
        DPODataset(file_path=fp, tokenizer=tokenizer, max_seq_len=0)
    with pytest.raises(ValueError, match="max_seq_len"):
        DPODataset(file_path=fp, tokenizer=tokenizer, max_seq_len=-1)
