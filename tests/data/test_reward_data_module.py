"""Tests for RewardDataModule contract compliance."""

import json

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
