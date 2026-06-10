"""Streaming datasets for large-scale language modeling."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

from llm.data.sources import TextSource
from llm.tokenization.tokenizer import BaseTokenizer


class StreamingTextDataset(IterableDataset):
    """
    Memory-efficient IterableDataset backed by a pluggable TextSource.

    Shards data across DDP ranks and DataLoader workers to avoid duplication.
    """

    def __init__(
        self,
        text_source: TextSource,
        tokenizer: BaseTokenizer,
        max_seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        overlap: int = 0,
        padding_value: int | None = None,
    ):
        self.text_source = text_source
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.rank = rank
        self.world_size = world_size
        self.overlap = overlap
        self.padding_value = padding_value if padding_value is not None else getattr(tokenizer, "pad_token_id", 0)

        if overlap >= max_seq_len:
            raise ValueError("overlap must be smaller than max_seq_len")

    def _shard_id(self) -> tuple[int, int]:
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        shard_id = self.rank * num_workers + worker_id
        num_shards = self.world_size * num_workers
        return shard_id, num_shards

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        shard_id, num_shards = self._shard_id()
        token_buffer: list[int] = []

        for line_idx, line in enumerate(self.text_source.iter_texts()):
            if line_idx % num_shards != shard_id:
                continue

            token_buffer.extend(self.tokenizer.encode(line))
            while len(token_buffer) >= self.max_seq_len:
                chunk = token_buffer[: self.max_seq_len]
                token_buffer = token_buffer[self.max_seq_len - self.overlap :] if self.overlap > 0 else []

                input_ids = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": input_ids, "labels": input_ids.clone()}
