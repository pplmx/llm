from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler

from llm.data.data_module import BaseDataModule
from llm.data.sft_dataset import SFTDataset
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.tokenization.tokenizer import BaseTokenizer, HFTokenizer


class SFTDataModule(BaseDataModule):
    """
    DataModule for Supervised Fine-Tuning (SFT) using SFTDataset.
    """

    def __init__(self, config: Any):
        super().__init__(config)
        self.tokenizer: BaseTokenizer | None = None
        self.train_dataset: SFTDataset | None = None
        self.val_dataset: SFTDataset | None = None

    def prepare_data(self):
        """
        Download tokenizer if needed.
        """
        data_config = self.config.data
        if data_config.tokenizer_type == "hf" and data_config.tokenizer_path:
            HFTokenizer.from_pretrained(data_config.tokenizer_path)

    def setup(self, stage: str | None = None):
        """
        Load tokenizer and create datasets.
        """
        data_config = self.config.data

        # 1. Load Tokenizer
        if data_config.tokenizer_type == "hf":
            if not data_config.tokenizer_path:
                raise ValueError("tokenizer_path must be specified for HF tokenizer.")
            self.tokenizer = HFTokenizer.from_pretrained(data_config.tokenizer_path)
        elif data_config.tokenizer_path and Path(data_config.tokenizer_path).exists():
            self.tokenizer = torch.load(data_config.tokenizer_path, weights_only=False)
        else:
            # Fallback
            corpus = ["<PAD>", "<EOS>", "<BOS>"]
            self.tokenizer = SimpleCharacterTokenizer(corpus)

        # 2. Create Datasets
        if not data_config.dataset_path:
            return

        # Assuming dataset_path is the jsonl file
        full_dataset = SFTDataset(
            file_path=data_config.dataset_path,
            tokenizer=self.tokenizer,
            max_seq_len=self.config.data.max_seq_len,
            # template_fn can be added based on config if needed, defaulting to Alpaca
        )

        # Split
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler]:
        if self.train_dataset is None:
            raise ValueError("Train dataset not initialized.")

        train_sampler = DistributedSampler(
            self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )

        use_persistent_workers = (
            self.config.optimization.persistent_workers and self.config.optimization.num_workers > 0
        )

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=train_sampler,
            num_workers=self.config.optimization.num_workers,
            pin_memory=self.config.optimization.pin_memory and torch.cuda.is_available(),
            persistent_workers=use_persistent_workers,
        )
        return loader, train_sampler

    def val_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler]:
        if self.val_dataset is None:
            return None, None

        val_sampler = DistributedSampler(
            self.val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )

        use_persistent_workers = (
            self.config.optimization.persistent_workers and self.config.optimization.num_workers > 0
        )

        loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            sampler=val_sampler,
            num_workers=self.config.optimization.num_workers,
            pin_memory=self.config.optimization.pin_memory and torch.cuda.is_available(),
            persistent_workers=use_persistent_workers,
        )
        return loader, val_sampler
