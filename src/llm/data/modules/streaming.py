"""Streaming DataModule for large-scale language modeling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler

from llm.data.base import StreamDataModule
from llm.data.datasets.streaming import StreamingTextDataset
from llm.data.datasets.text import TextDataset
from llm.data.sources import build_text_source
from llm.data.stream_state import StreamDataState
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.tokenization.tokenizer import BaseTokenizer, HFTokenizer


class StreamingTextDataModule(StreamDataModule):
    """Iterable DataModule for memory-bounded pretraining."""

    def __init__(self, config: Any):
        super().__init__(config)
        self.tokenizer: BaseTokenizer | None = None
        self.train_dataset: StreamingTextDataset | None = None
        self.val_dataset: TextDataset | None = None
        self.stream_data_state = StreamDataState()

    def prepare_data(self):
        data_config = self.config.data
        if data_config.tokenizer_type == "hf" and data_config.tokenizer_path:
            HFTokenizer.from_pretrained(data_config.tokenizer_path)

    def setup(self, stage: str | None = None):
        self.validate_streaming_config()
        self.tokenizer = self._load_tokenizer()

        text_source = build_text_source(self.config.data)
        self.train_dataset = StreamingTextDataset(
            text_source=text_source,
            tokenizer=self.tokenizer,
            max_seq_len=self.config.data.max_seq_len,
            stream_data_state=self.stream_data_state,
        )

        val_path = self.config.data.val_dataset_path
        if val_path:
            self.val_dataset = TextDataset(
                file_path=val_path,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.data.max_seq_len,
            )

    def _load_tokenizer(self) -> BaseTokenizer:
        data_config = self.config.data
        if data_config.tokenizer_type == "hf":
            if not data_config.tokenizer_path:
                raise ValueError("tokenizer_path must be specified for HF tokenizer.")
            return HFTokenizer.from_pretrained(data_config.tokenizer_path)

        if data_config.tokenizer_path and Path(data_config.tokenizer_path).exists():
            return torch.load(data_config.tokenizer_path, weights_only=False)

        corpus = ["<PAD>", "<EOS>", "<BOS>"]
        return SimpleCharacterTokenizer(corpus)

    def train_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, None]:
        if self.train_dataset is None:
            raise ValueError("Train dataset not initialized.")

        self.train_dataset.rank = rank
        self.train_dataset.world_size = world_size

        use_persistent_workers = (
            self.config.optimization.persistent_workers and self.config.optimization.num_workers > 0
        )
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.optimization.num_workers,
            pin_memory=self.config.optimization.pin_memory and torch.cuda.is_available(),
            persistent_workers=use_persistent_workers,
        )
        return loader, None

    def val_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader | None, DistributedSampler | None]:
        if self.val_dataset is None:
            return None, None

        val_sampler = DistributedSampler(
            self.val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            sampler=val_sampler,
            num_workers=self.config.optimization.num_workers,
            pin_memory=self.config.optimization.pin_memory and torch.cuda.is_available(),
        )
        return loader, val_sampler

    def get_checkpoint_state(self) -> dict | None:
        return {"stream_data": self.stream_data_state.to_dict()}

    def load_checkpoint_state(self, state: dict | None) -> None:
        if not state:
            return
        self.stream_data_state = StreamDataState.from_dict(state.get("stream_data"))
        if self.train_dataset is not None:
            self.train_dataset.stream_data_state = self.stream_data_state
