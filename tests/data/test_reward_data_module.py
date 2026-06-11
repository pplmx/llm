"""Tests for RewardDataModule contract compliance."""

import json
from string import printable

from llm.data.datasets.reward import RewardDataset
from llm.data.modules.reward import RewardDataModule
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.training.core.config import Config


def _write_reward_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _build_data_module(config: Config) -> RewardDataModule:
    data_module = RewardDataModule(config)
    data_module.prepare_data()
    data_module.tokenizer = SimpleCharacterTokenizer(corpus=[printable])

    if not config.data.dataset_path:
        return data_module

    full_dataset = RewardDataset(
        file_path=config.data.dataset_path,
        tokenizer=data_module.tokenizer,
        max_seq_len=config.data.max_seq_len,
    )
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    if val_size > 0:
        from torch.utils.data import random_split

        data_module.train_dataset, data_module.val_dataset = random_split(full_dataset, [train_size, val_size])
    else:
        data_module.train_dataset = full_dataset
        data_module.val_dataset = None

    return data_module


def test_reward_data_module_uses_base_contract(tmp_path):
    data_file = tmp_path / "reward.jsonl"
    _write_reward_jsonl(
        data_file,
        [
            {"prompt": "Q1", "chosen": "A1", "rejected": "B1"},
            {"prompt": "Q2", "chosen": "A2", "rejected": "B2"},
            {"prompt": "Q3", "chosen": "A3", "rejected": "B3"},
        ],
    )

    config = Config()
    config.data.dataset_path = str(data_file)
    config.data.max_seq_len = 32
    config.training.batch_size = 2
    config.optimization.num_workers = 0

    data_module = _build_data_module(config)

    train_loader, train_sampler = data_module.train_dataloader(rank=0, world_size=1)
    val_loader, val_sampler = data_module.val_dataloader(rank=0, world_size=1)

    assert train_sampler is not None
    assert val_sampler is not None
    batch = next(iter(train_loader))
    assert "chosen_input_ids" in batch
    assert "rejected_input_ids" in batch
    assert batch["chosen_input_ids"].shape[0] == 2

    if val_loader is not None:
        val_batch = next(iter(val_loader))
        assert val_batch["chosen_input_ids"].shape[0] <= 2


def test_reward_data_module_explicit_val_file(tmp_path):
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    _write_reward_jsonl(train_file, [{"prompt": "Q", "chosen": "A", "rejected": "B"}])
    _write_reward_jsonl(val_file, [{"prompt": "Q2", "chosen": "C", "rejected": "D"}])

    config = Config()
    config.data.dataset_path = str(train_file)
    config.data.val_dataset_path = str(val_file)
    config.data.max_seq_len = 16
    config.training.batch_size = 1
    config.optimization.num_workers = 0

    data_module = RewardDataModule(config)
    data_module.setup()

    assert isinstance(data_module.train_dataset, RewardDataset)
    assert len(data_module.train_dataset) == 1
    assert len(data_module.val_dataset) == 1
