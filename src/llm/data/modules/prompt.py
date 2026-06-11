"""Prompt DataModule for PPO rollouts."""

from __future__ import annotations

from typing import Any

from llm.data.base import MapDataModule
from llm.data.datasets.prompt import PromptDataset
from llm.data.modules.map_base import TokenizedMapDataModule


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

        full_dataset = PromptDataset(data_config.dataset_path)
        if data_config.val_dataset_path:
            self.train_dataset = full_dataset
            self.val_dataset = PromptDataset(data_config.val_dataset_path)
        else:
            self.train_dataset, self.val_dataset = TokenizedMapDataModule.split_train_val(full_dataset)

    def train_dataloader(self, rank: int, world_size: int):
        if self.train_dataset is None:
            raise ValueError("Train dataset not initialized.")

        from torch.utils.data import DistributedSampler

        sampler = DistributedSampler(
            self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        loader = TokenizedMapDataModule.build_dataloader(
            self,
            self.train_dataset,
            sampler,
            collate_fn=collate_prompts,
        )
        return loader, sampler

    def val_dataloader(self, rank: int, world_size: int):
        if self.val_dataset is None:
            return None, None

        from torch.utils.data import DistributedSampler

        sampler = DistributedSampler(
            self.val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        loader = TokenizedMapDataModule.build_dataloader(
            self,
            self.val_dataset,
            sampler,
            collate_fn=collate_prompts,
        )
        return loader, sampler
