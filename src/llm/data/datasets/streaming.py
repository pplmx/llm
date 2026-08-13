"""Streaming datasets for large-scale language modeling."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

from llm.data.sources import TextSource
from llm.data.stream_state import StreamDataState
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
        stream_data_state: StreamDataState | None = None,
    ):
        self.text_source = text_source
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.rank = rank
        self.world_size = world_size
        self.overlap = overlap
        self.padding_value = padding_value if padding_value is not None else getattr(tokenizer, "pad_token_id", 0)
        self.stream_data_state = stream_data_state or StreamDataState()

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

    def _worker_id_and_count(self) -> tuple[int, int]:
        worker_info = get_worker_info()
        if worker_info is None:
            return 0, 1
        return worker_info.id, worker_info.num_workers

    def reset(self) -> None:
        """Clear the resume cursor so the next iteration restarts the corpus.

        Called by the training engine when the streaming source is exhausted
        before ``steps_per_epoch`` is reached: pretraining cycles the corpus
        (optionally de-duplicated) until the step budget is met.

        If the underlying source is a persistent-dedup wrapper, its
        cross-run seen-set is also cleared: otherwise a corpus whose whole
        content was consumed+hashed in a prior run classifies every record
        as already-seen on the recycled pass and the engine raises
        ``"streaming corpus is empty"`` (RIL ISS-064). In-memory per-pass
        dedup is unaffected.
        """
        self.stream_data_state.reset()
        reset_cross_run = getattr(self.text_source, "reset_cross_run_seen", None)
        if reset_cross_run is not None:
            reset_cross_run()

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        shard_id, num_shards = self._shard_id()
        worker_id, num_workers = self._worker_id_and_count()
        state = self.stream_data_state.get_shard(self.rank, worker_id, num_workers)
        token_buffer = list(state.token_buffer)

        for line_idx, line in enumerate(
            self.text_source.iter_texts(skip=state.line_index),
            start=state.line_index,
        ):
            if line_idx % num_shards != shard_id:
                state.line_index = line_idx + 1
                continue

            token_buffer.extend(self.tokenizer.encode(line))
            state.line_index = line_idx + 1
            state.token_buffer = token_buffer

            while len(token_buffer) >= self.max_seq_len:
                chunk = token_buffer[: self.max_seq_len]
                token_buffer = token_buffer[self.max_seq_len - self.overlap :] if self.overlap > 0 else []
                state.token_buffer = token_buffer

                input_ids = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": input_ids, "labels": input_ids.clone()}
