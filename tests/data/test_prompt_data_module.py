"""Tests for PromptDataModule — PPO prompt data loading."""

import json

import pytest

from llm.data.modules.prompt import PromptDataModule
from llm.training.core.config import Config, DataConfig, ModelConfig, OptimizationConfig, TrainingConfig


def _make_config(data_config: DataConfig) -> Config:
    """Build a Config with the given DataConfig."""
    return Config(
        model=ModelConfig(hidden_size=16, num_layers=1),
        training=TrainingConfig(task="ppo", epochs=1, batch_size=2, num_samples=10),
        optimization=OptimizationConfig(num_workers=0),
        data=data_config,
    )


def _write_prompts(tmp_path, prompts: list[str], filename: str = "prompts.jsonl") -> str:
    """Write prompt data to a JSONL file and return the path."""
    path = tmp_path / filename
    with path.open("w") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}) + "\n")
    return str(path)


def test_prompt_module_setup(tmp_path):
    """setup() loads train_dataset and splits validation."""
    data_file = _write_prompts(tmp_path, ["Hello", "World", "Test", "Foo", "Bar"])
    config = _make_config(DataConfig(dataset_path=data_file))

    module = PromptDataModule(config)
    module.setup()

    assert module.train_dataset is not None
    assert len(module.train_dataset) <= 5  # 90% of 5 = 4 or 5
    assert module.val_dataset is not None
    assert len(module.val_dataset) >= 0


def test_prompt_module_setup_with_val_path(tmp_path):
    """When val_dataset_path is provided, both datasets are loaded from their own files."""
    train_file = _write_prompts(tmp_path, ["Hello", "World", "Train"], "train.jsonl")
    val_file = _write_prompts(tmp_path, ["Val1", "Val2"], "val.jsonl")
    config = _make_config(DataConfig(dataset_path=train_file, val_dataset_path=val_file))

    module = PromptDataModule(config)
    module.setup()

    assert module.train_dataset is not None
    assert len(module.train_dataset) == 3
    assert module.val_dataset is not None
    assert len(module.val_dataset) == 2


def test_prompt_module_setup_without_dataset_path(tmp_path):
    """setup() raises ValueError when dataset_path is not set."""
    config = _make_config(DataConfig(dataset_path=None))

    module = PromptDataModule(config)
    with pytest.raises(ValueError, match="dataset_path is required"):
        module.setup()


def test_prompt_module_train_dataloader_not_initialized(tmp_path):
    """train_dataloader raises ValueError when setup() was not called."""
    data_file = _write_prompts(tmp_path, ["Hello"])
    config = _make_config(DataConfig(dataset_path=data_file))

    module = PromptDataModule(config)
    # setup() not called — train_dataset is None
    with pytest.raises(ValueError, match="Train dataset not initialized"):
        module.train_dataloader(rank=0, world_size=1)


def test_prompt_module_val_dataloader_not_initialized(tmp_path):
    """val_dataloader returns (None, None) when setup() was not called."""
    data_file = _write_prompts(tmp_path, ["Hello"])
    config = _make_config(DataConfig(dataset_path=data_file))

    module = PromptDataModule(config)
    # setup() not called — val_dataset is None
    loader, sampler = module.val_dataloader(rank=0, world_size=1)
    assert loader is None
    assert sampler is None


def test_prompt_module_train_dataloader_returns_loader(tmp_path):
    """train_dataloader returns (DataLoader, DistributedSampler)."""
    data_file = _write_prompts(tmp_path, ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
    config = _make_config(DataConfig(dataset_path=data_file))

    module = PromptDataModule(config)
    module.setup()

    loader, sampler = module.train_dataloader(rank=0, world_size=1)
    assert loader is not None
    assert sampler is not None


def test_prompt_module_collate_function(tmp_path):
    """verify prompt_data_collate function works correctly."""
    from llm.data.modules.prompt import collate_prompts

    batch = [{"prompt": "Hello"}, {"prompt": "World"}]
    result = collate_prompts(batch)
    assert result == {"prompts": ["Hello", "World"]}


def test_prompt_module_val_dataloader_returns_loader(tmp_path):
    """val_dataloader returns a real loader after setup()."""
    data_file = _write_prompts(tmp_path, ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
    config = _make_config(DataConfig(dataset_path=data_file))

    module = PromptDataModule(config)
    module.setup()

    loader, sampler = module.val_dataloader(rank=0, world_size=1)
    assert loader is not None
    assert sampler is not None
