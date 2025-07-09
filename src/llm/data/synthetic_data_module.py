from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from llm.data.data_module import BaseDataModule


class SyntheticDataModule(BaseDataModule):
    """
    A DataModule for generating synthetic regression data.
    """

    def __init__(self, config: Any):
        super().__init__(config)
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        # No data to download/prepare for synthetic data
        pass

    def setup(self, stage: str | None = None):
        # Create synthetic training dataset
        train_x = torch.randn(self.config.training.num_samples, self.config.model.hidden_size)
        train_y = train_x + 0.1 * torch.randn_like(train_x)
        self.train_dataset = TensorDataset(train_x, train_y)

        # Create synthetic validation dataset (e.g., 10% of training size)
        val_num_samples = max(1, self.config.training.num_samples // 10)
        val_x = torch.randn(val_num_samples, self.config.model.hidden_size)
        val_y = val_x + 0.1 * torch.randn_like(val_x)
        self.val_dataset = TensorDataset(val_x, val_y)

    def train_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler]:
        train_sampler = DistributedSampler(
            self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        use_persistent_workers = (
            self.config.optimization.persistent_workers and self.config.optimization.num_workers > 0
        )

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=train_sampler,
            num_workers=self.config.optimization.num_workers,
            pin_memory=self.config.optimization.pin_memory,
            prefetch_factor=self.config.optimization.prefetch_factor if use_persistent_workers else 2,
            persistent_workers=use_persistent_workers,
        )
        return train_dataloader, train_sampler

    def val_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler]:
        val_sampler = DistributedSampler(
            self.val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        use_persistent_workers = (
            self.config.optimization.persistent_workers and self.config.optimization.num_workers > 0
        )

        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            sampler=val_sampler,
            num_workers=self.config.optimization.num_workers,
            pin_memory=self.config.optimization.pin_memory,
            prefetch_factor=self.config.optimization.prefetch_factor if use_persistent_workers else 2,
            persistent_workers=use_persistent_workers,
        )
        return val_dataloader, val_sampler
