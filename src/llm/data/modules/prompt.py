"""Prompt DataModule for PPO rollouts."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, DistributedSampler, random_split

from llm.data.base import MapDataModule
from llm.data.datasets.prompt import PromptDataset


def collate_prompts(batch: list[dict[str, str]]) -> dict[str, list[str]]:
    return {"prompts": [item["prompt"] for item in batch]}


class PromptDataModule(MapDataModule):
    """DataModule that yields prompt batches for PPO rollouts."""

    def __init__(self, config: Any):
        super().__init__(config)
        self.train_dataset: PromptDataset | None = None
        self.val_dataset: PromptDataset | None = None

    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None):
        data_config = self.config.data
        if not data_config.dataset_path:
            raise ValueError("data.dataset_path is required for PPO prompt data.")

        self.train_dataset = PromptDataset(data_config.dataset_path)

        val_path = data_config.val_dataset_path
        if val_path:
            self.val_dataset = PromptDataset(val_path)
        else:
            train_size = int(0.9 * len(self.train_dataset))
            val_size = len(self.train_dataset) - train_size
            if val_size > 0:
                self.train_dataset, val_dataset = random_split(self.train_dataset, [train_size, val_size])
                self.val_dataset = val_dataset
            else:
                self.val_dataset = None

    def train_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler | None]:
        if self.train_dataset is None:
            raise ValueError("Train dataset not initialized.")

        sampler = DistributedSampler(
            self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=sampler,
            collate_fn=collate_prompts,
            num_workers=self.config.optimization.num_workers,
            pin_memory=self.config.optimization.pin_memory,
        )
        return loader, sampler

    def val_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader | None, DistributedSampler | None]:
        if self.val_dataset is None:
            return None, None

        sampler = DistributedSampler(
            self.val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            sampler=sampler,
            collate_fn=collate_prompts,
            num_workers=self.config.optimization.num_workers,
            pin_memory=self.config.optimization.pin_memory,
        )
        return loader, sampler
