import json
from string import printable

import torch

from llm.data.dpo_dataset import DPODataset
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


def test_dpo_dataset(tmp_path):
    data = [{"prompt": "Q:", "chosen": "Good", "rejected": "Bad"}]
    file_path = tmp_path / "dpo_data.jsonl"
    with file_path.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    tokenizer = SimpleCharacterTokenizer([printable])

    dataset = DPODataset(file_path=file_path, tokenizer=tokenizer, max_seq_len=20)

    assert len(dataset) == 1
    item = dataset[0]

    assert "chosen_input_ids" in item
    assert "rejected_input_ids" in item
    assert "chosen_labels" in item

    # Verify content
    # Prompt "Q:"
    # Chosen "Good"
    # Rejected "Bad"

    # Check Chosen
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
