"""Reward Model DataModule for RLHF."""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler, random_split

from llm.data.base import MapDataModule
from llm.data.datasets.reward import RewardDataset
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.tokenization.tokenizer import BaseTokenizer, HFTokenizer


class RewardDataModule(MapDataModule):
    """DataModule for Reward Model training with DDP-compatible loaders."""

    def __init__(self, config: Any):
        super().__init__(config)
        self.tokenizer: BaseTokenizer | None = None
        self.train_dataset: RewardDataset | None = None
        self.val_dataset: RewardDataset | None = None

    def prepare_data(self):
        data_config = self.config.data
        if data_config.tokenizer_type == "hf" and data_config.tokenizer_path:
            HFTokenizer.from_pretrained(data_config.tokenizer_path)

    def setup(self, stage: str | None = None):
        data_config = self.config.data

        if data_config.tokenizer_type == "hf":
            if not data_config.tokenizer_path:
                raise ValueError("tokenizer_path must be specified for HF tokenizer.")
            self.tokenizer = HFTokenizer.from_pretrained(data_config.tokenizer_path)
        elif data_config.tokenizer_path and Path(data_config.tokenizer_path).exists():
            self.tokenizer = torch.load(data_config.tokenizer_path, weights_only=False)
        else:
            corpus = ["<PAD>", "<EOS>", "<BOS>"]
            self.tokenizer = SimpleCharacterTokenizer(corpus)

        if not data_config.dataset_path:
            return

        full_dataset = RewardDataset(
            file_path=data_config.dataset_path,
            tokenizer=self.tokenizer,
            max_seq_len=data_config.max_seq_len,
        )

        val_path = getattr(data_config, "val_dataset_path", None)
        if val_path:
            self.train_dataset = full_dataset
            self.val_dataset = RewardDataset(
                file_path=val_path,
                tokenizer=self.tokenizer,
                max_seq_len=data_config.max_seq_len,
            )
        else:
            train_size = int(0.9 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            if val_size > 0:
                self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
            else:
                self.train_dataset = full_dataset
                self.val_dataset = None

    def train_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler | None]:
        if self.train_dataset is None:
            raise ValueError("Train dataset not initialized.")

        train_sampler = DistributedSampler(
            self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        loader = self._build_dataloader(self.train_dataset, train_sampler)
        return loader, train_sampler

    def val_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader | None, DistributedSampler | None]:
        if self.val_dataset is None:
            return None, None

        val_sampler = DistributedSampler(
            self.val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        loader = self._build_dataloader(self.val_dataset, val_sampler)
        return loader, val_sampler

    def _build_dataloader(self, dataset, sampler: DistributedSampler) -> DataLoader:
        use_persistent_workers = (
            self.config.optimization.persistent_workers and self.config.optimization.num_workers > 0
        )
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            sampler=sampler,
            num_workers=self.config.optimization.num_workers,
            pin_memory=self.config.optimization.pin_memory and torch.cuda.is_available(),
            persistent_workers=use_persistent_workers,
        )
