"""Tests for RewardDataModule contract compliance."""

import json

import pytest
import torch

from llm.data.modules.reward import RewardDataModule
from llm.runtime.tokenizer_factory import TokenizerFactory
from llm.training.core.config import Config


def _write_reward_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _config_with_printable_tokenizer(tmp_path, *, train_path: str, val_path: str | None = None) -> Config:
    tokenizer_path = tmp_path / "tokenizer.pt"
    torch.save(TokenizerFactory.from_printable_corpus(), tokenizer_path)

    config = Config()
    config.data.dataset_path = train_path
    config.data.val_dataset_path = val_path
    config.data.tokenizer_path = str(tokenizer_path)
    config.data.max_seq_len = 32
    config.training.batch_size = 2
    config.optimization.num_workers = 0
    return config


def test_reward_data_module_setup_splits_train_and_val(tmp_path):
    """Requirement: RewardDataModule.setup applies 90/10 split for three preference pairs."""
    data_file = tmp_path / "reward.jsonl"
    _write_reward_jsonl(
        data_file,
        [
            {"prompt": "Q1", "chosen": "A1", "rejected": "B1"},
            {"prompt": "Q2", "chosen": "A2", "rejected": "B2"},
            {"prompt": "Q3", "chosen": "A3", "rejected": "B3"},
        ],
    )

    data_module = RewardDataModule(_config_with_printable_tokenizer(tmp_path, train_path=str(data_file)))
    data_module.setup()

    assert len(data_module.train_dataset) == 2
    assert len(data_module.val_dataset) == 1

    train_loader, train_sampler = data_module.train_dataloader(rank=0, world_size=1)
    val_loader, _ = data_module.val_dataloader(rank=0, world_size=1)

    assert len(train_sampler) == 2
    assert len(val_loader.dataset) == 1
    batch = next(iter(train_loader))
    assert batch["chosen_input_ids"].shape == (2, 32)
    assert batch["rejected_input_ids"].shape == (2, 32)


def test_reward_data_module_explicit_val_file(tmp_path):
    """Requirement: val_dataset_path loads validation JSONL without random split."""
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    _write_reward_jsonl(train_file, [{"prompt": "Q", "chosen": "A", "rejected": "B"}])
    _write_reward_jsonl(val_file, [{"prompt": "Q2", "chosen": "C", "rejected": "D"}])

    data_module = RewardDataModule(
        _config_with_printable_tokenizer(
            tmp_path,
            train_path=str(train_file),
            val_path=str(val_file),
        )
    )
    data_module.setup()

    assert len(data_module.train_dataset) == 1
    assert len(data_module.val_dataset) == 1
    assert data_module.train_dataset.data[0]["prompt"] == "Q"
    assert data_module.val_dataset.data[0]["prompt"] == "Q2"


def test_reward_dataset_truncation_keeps_response_tail(tmp_path):
    """Truncation drops the PROMPT PREFIX, preserving the response END — the
    position the reward model actually scores (RIL ISS-332).

    The old ``[:max_seq_len]`` kept the prompt and chopped the response tail,
    so the scored last non-pad token was an arbitrary mid-response token.
    """
    import json
    from string import printable

    from llm.data.datasets.reward import RewardDataset
    from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer

    tok = SimpleCharacterTokenizer([printable])
    fp = tmp_path / "long.jsonl"
    fp.write_text(
        json.dumps({"prompt": "P:", "chosen": "R" * 80, "rejected": "Bad"}) + "\n",
        encoding="utf-8",
    )

    dataset = RewardDataset(file_path=fp, tokenizer=tok, max_seq_len=6)
    item = dataset[0]

    prompt_ids = tok.encode("P:")
    chosen_ids = tok.encode("R" * 80)
    combined = prompt_ids + chosen_ids
    assert item["chosen_input_ids"].tolist() == combined[-6:]
    # The response end survived — the whole window is real tokens, mask all 1s.
    assert item["chosen_attention_mask"].tolist() == [1] * 6


def test_reward_dataset_rejects_nonpositive_max_seq_len(tmp_path):
    """RIL ISS-199: a non-positive ``max_seq_len`` fails fast instead of
    silently producing misaligned attention_mask/input_ids."""
    from string import printable

    from llm.data.datasets.reward import RewardDataset
    from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer

    tok = SimpleCharacterTokenizer([printable])
    fp = tmp_path / "x.jsonl"
    fp.write_text('{"prompt":"Q:","chosen":"Good","rejected":"Bad"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="max_seq_len"):
        RewardDataset(file_path=fp, tokenizer=tok, max_seq_len=0)
    with pytest.raises(ValueError, match="max_seq_len"):
        RewardDataset(file_path=fp, tokenizer=tok, max_seq_len=-1)


def test_reward_dataset_json_decode_error(tmp_path):
    """RIL ISS-201: a malformed JSONL line raises an actionable ValueError
    with the file path, aligned with SFT/DPO — not a raw JSONDecodeError."""
    from string import printable

    from llm.data.datasets.reward import RewardDataset
    from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer

    tok = SimpleCharacterTokenizer([printable])
    fp = tmp_path / "bad_json.jsonl"
    fp.write_text("not valid json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        RewardDataset(file_path=fp, tokenizer=tok, max_seq_len=20)


def test_reward_dataset_drops_empty_completion(tmp_path):
    """An empty chosen/rejected completion makes the reward model score the
    prompt itself (or a fully-masked row) instead of the response end — the
    row must be dropped at load (RIL ISS-336)."""
    from string import printable

    from llm.data.datasets.reward import RewardDataset
    from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer

    tok = SimpleCharacterTokenizer([printable])
    fp = tmp_path / "empty.jsonl"
    fp.write_text(
        '{"prompt":"Q","chosen":"","rejected":"Bad"}\n{"prompt":"Q2","chosen":"Good","rejected":"Bad"}\n',
        encoding="utf-8",
    )

    dataset = RewardDataset(file_path=fp, tokenizer=tok, max_seq_len=20)
    assert len(dataset) == 1
    assert dataset.data[0]["prompt"] == "Q2"
