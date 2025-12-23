from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler

from llm.data.data_module import BaseDataModule
from llm.data.loader import TextDataset
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.tokenization.tokenizer import BaseTokenizer, HFTokenizer


class TextDataModule(BaseDataModule):
    """
    DataModule for Language Modeling using TextDataset.
    """

    def __init__(self, config: Any):
        super().__init__(config)
        self.tokenizer: BaseTokenizer | None = None
        self.train_dataset: TextDataset | None = None
        self.val_dataset: TextDataset | None = None

    def prepare_data(self):
        """
        Download data or tokenizer if needed.
        """
        # In multi-node, this is called only on rank 0.
        # Ensure tokenizer is downloadable/cached.
        # Ensure tokenizer is downloadable/cached.
        data_config = self.config.data
        if data_config.tokenizer_type == "hf" and data_config.tokenizer_path:
            HFTokenizer.from_pretrained(data_config.tokenizer_path)

    def setup(self, stage: str | None = None):
        """
        Load tokenizer and create datasets.
        Called on every GPU.
        """
        data_config = self.config.data

        # 1. Load Tokenizer
        if data_config.tokenizer_type == "hf":
            if not data_config.tokenizer_path:
                raise ValueError("tokenizer_path must be specified for HF tokenizer.")
            self.tokenizer = HFTokenizer.from_pretrained(data_config.tokenizer_path)
        else:
            # Simple tokenizer needs a corpus.
            # For simplicity, if we are in simple mode, we might need to build vocab from dataset
            # or load an existing tokenizer file (not implemented for simple yet except via pickle).
            # Here we assume tokenizer_path points to a pickle or we fail.
            # OR we build it from the text file on the fly (slow/inconsistent across ranks if not careful).
            # For now, let's assume simple mode expects a pickle path if provided, or we fail.
            if data_config.tokenizer_path and Path(data_config.tokenizer_path).exists():
                self.tokenizer = torch.load(data_config.tokenizer_path, weights_only=False)
            else:
                # Fallback: Build from dataset_path (only on rank 0? No, setup is per process)
                # This is tricky for DDP.
                # We'll raise error for 'simple' if not pre-saved, to encourage 'hf'.
                pass

        if self.tokenizer is None:
            # Basic fallback for testing if no path
            corpus = ["<PAD>", "<EOS>", "<BOS>"]  # Minimal
            self.tokenizer = SimpleCharacterTokenizer(corpus)

        # 2. Create Datasets
        if not data_config.dataset_path:
            # No dataset provided, maybe dummy?
            return

        # Assuming dataset_path is a text file.
        # We might split it into train/val.
        full_dataset = TextDataset(
            file_path=data_config.dataset_path,
            tokenizer=self.tokenizer,
            max_seq_len=self.config.data.max_seq_len,
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
            # Return empty if no val set
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
