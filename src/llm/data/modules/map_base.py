"""Shared helpers for map-style DataModules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, random_split

from llm.data.base import MapDataModule
from llm.runtime.tokenizer_factory import TokenizerFactory
from llm.tokenization.tokenizer import BaseTokenizer


class SamplerMapDataModule(MapDataModule):
    """Map DataModule with shared DistributedSampler DataLoader helpers."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

    def build_dataloader(
        self,
        dataset: Dataset,
        sampler: DistributedSampler,
        *,
        collate_fn=None,
    ) -> DataLoader:
        optimization = self.config.optimization
        use_persistent_workers = optimization.persistent_workers and optimization.num_workers > 0
        kwargs: dict[str, Any] = {
            "batch_size": self.config.training.batch_size,
            "sampler": sampler,
            "num_workers": optimization.num_workers,
            "pin_memory": optimization.pin_memory and torch.cuda.is_available(),
            "persistent_workers": use_persistent_workers,
        }
        if optimization.num_workers > 0:
            kwargs["prefetch_factor"] = optimization.prefetch_factor
        if collate_fn is not None:
            kwargs["collate_fn"] = collate_fn
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler | None]:
        if self.train_dataset is None:
            raise ValueError("Train dataset not initialized.")

        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        return self.build_dataloader(self.train_dataset, train_sampler), train_sampler

    def val_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader | None, DistributedSampler | None]:
        if self.val_dataset is None:
            return None, None

        val_sampler = DistributedSampler(
            self.val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        return self.build_dataloader(self.val_dataset, val_sampler), val_sampler


class TokenizedMapDataModule(SamplerMapDataModule):
    """Map DataModule with shared tokenizer loading and DistributedSampler loaders."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.tokenizer: BaseTokenizer | None = None

    def prepare_data(self) -> None:
        TokenizerFactory.cache_hf_tokenizer(self.config.data)

    def setup_tokenizer(self) -> None:
        self.tokenizer = TokenizerFactory.from_data_config(self.config.data)

    @staticmethod
    def split_train_val(dataset: Dataset, train_ratio: float = 0.9) -> tuple[Dataset, Dataset | None]:
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        if val_size <= 0:
            return dataset, None
        return random_split(dataset, [train_size, val_size])

    def assign_train_val_datasets(
        self,
        full_dataset: Dataset,
        *,
        val_path: str | None = None,
        build_val_dataset: Callable[[str], Dataset] | None = None,
        train_ratio: float = 0.9,
    ) -> None:
        """Assign train/val datasets from a full dataset or explicit val path."""
        if val_path:
            if build_val_dataset is None:
                raise ValueError("build_val_dataset is required when val_path is set.")
            self.train_dataset = full_dataset
            self.val_dataset = build_val_dataset(val_path)
            return

        self.train_dataset, self.val_dataset = self.split_train_val(full_dataset, train_ratio)

    def setup_tokenized_file_dataset(self, dataset_cls: type, stage: str | None = None) -> None:
        """Shared setup for file-backed tokenized datasets."""
        self.setup_tokenizer()
        data_config = self.config.data
        if not data_config.dataset_path:
            return

        full_dataset = dataset_cls(
            file_path=data_config.dataset_path,
            tokenizer=self.tokenizer,
            max_seq_len=data_config.max_seq_len,
        )
        self.assign_train_val_datasets(
            full_dataset,
            val_path=data_config.val_dataset_path,
            build_val_dataset=lambda path: dataset_cls(
                file_path=path,
                tokenizer=self.tokenizer,
                max_seq_len=data_config.max_seq_len,
            ),
        )
