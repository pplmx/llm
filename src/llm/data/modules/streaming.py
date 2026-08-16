"""Streaming DataModule for large-scale language modeling."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from llm.data.base import StreamDataModule
from llm.data.datasets.streaming import StreamingTextDataset
from llm.data.datasets.text import TextDataset
from llm.data.sources import (
    DedupTextSource,
    build_text_source,
    source_fingerprint_from_config,
    validate_source_fingerprint,
)
from llm.data.stream_state import StreamDataState
from llm.runtime.tokenizer_factory import TokenizerFactory
from llm.tokenization.tokenizer import BaseTokenizer

logger = logging.getLogger(__name__)


class StreamingTextDataModule(StreamDataModule):
    """Iterable DataModule for memory-bounded pretraining."""

    def __init__(self, config: Any):
        super().__init__(config)
        self.tokenizer: BaseTokenizer | None = None
        self.train_dataset: StreamingTextDataset | None = None
        self.val_dataset: TextDataset | None = None
        self.stream_data_state = StreamDataState()
        # World size seen by ``train_dataloader``; stamped into checkpoints so
        # resume can reject layouts whose shard cursors are not interchangeable.
        self._world_size: int | None = None
        # World size carried by a checkpoint loaded BEFORE ``train_dataloader``
        # ran (and therefore before ``_world_size`` was known). The mismatch
        # check is deferred to ``train_dataloader`` instead of being silently
        # skipped (RIL ISS-204: the guard was order-dependent on
        # ``_world_size`` already being set when ``load_checkpoint_state``
        # ran — ``load_extra_state`` makes no such ordering guarantee).
        self._pending_world_size: int | None = None

    def prepare_data(self):
        TokenizerFactory.cache_hf_tokenizer(self.config.data)

    def setup(self, stage: str | None = None):
        self.validate_streaming_config()
        self.tokenizer = self._load_tokenizer()

        text_source = build_text_source(self.config.data)
        if isinstance(text_source, DedupTextSource) and not text_source.write_seen_hashes:
            logger.warning(
                "DedupTextSource is running with in-memory only state "
                "(write_seen_hashes=False). Checkpoint resume re-creates the source, "
                "so previously-deduplicated records will be re-processed after a resume. "
                "Set data.seen_hashes_path + data.write_seen_hashes=True for cross-run "
                "dedup consistency."
            )
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
        return TokenizerFactory.from_data_config(self.config.data)

    def _validate_world_size(self, saved_world_size: int | None) -> None:
        """Refuse a checkpoint whose shard cursors were built for another
        rank layout.

        Shard cursors depend on the ``rank % num_shards`` arithmetic, so a
        checkpoint saved under ``world_size=A`` is meaningless to a run using
        ``world_size=B`` (ranks silently re-train wrong shards). Called both
        from ``load_checkpoint_state`` (when ``_world_size`` is already known)
        and from ``train_dataloader`` (for a checkpoint loaded earlier —
        RIL ISS-204).
        """
        if saved_world_size is None or self._world_size is None:
            return
        if int(saved_world_size) != self._world_size:
            raise ValueError(
                "Streaming checkpoint was saved with world_size="
                f"{saved_world_size} but this run uses world_size={self._world_size}. "
                "Shard cursors depend on the rank layout and are not interchangeable "
                "across world sizes; resume with the same number of ranks."
            )

    def train_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, None]:
        if self.train_dataset is None:
            raise ValueError("Train dataset not initialized.")

        self.train_dataset.rank = rank
        self.train_dataset.world_size = world_size
        self._world_size = world_size
        # A checkpoint may have been loaded before the world size was known;
        # validate the deferred mismatch now that it is.
        self._validate_world_size(self._pending_world_size)
        self._pending_world_size = None

        optimization = self.config.optimization
        # The resume cursor (``stream_data_state``) lives on the dataset
        # object in the main process. DataLoader workers run on forked
        # copies, so their cursor mutations never reach the main process:
        # a checkpoint saved mid-run would lose all progress and resume
        # would silently re-read the corpus from the start. Streaming
        # therefore runs single-process so checkpoints capture the real
        # cursor.
        num_workers = optimization.num_workers
        if num_workers > 0:
            logger.warning(
                "StreamingTextDataset keeps its resume cursor in the main process; "
                "DataLoader workers fork it and their progress is lost at checkpoint "
                "time. Forcing num_workers=0 so checkpoint resume stays correct. "
                "Set optimization.num_workers=0 explicitly to silence this warning."
            )
            num_workers = 0

        use_persistent_workers = optimization.persistent_workers and num_workers > 0
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            num_workers=num_workers,
            pin_memory=optimization.pin_memory and torch.cuda.is_available(),
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
        shards = self.stream_data_state.to_dict()
        # Only rank 0 persists the checkpoint (CheckpointManager ignores other
        # ranks), so the saved state must carry EVERY rank's shard cursor —
        # otherwise resumed ranks without a saved shard silently restart from
        # line 0 and re-train the corpus. ``get_checkpoint_state`` is called
        # on all ranks by the engine, which makes this collective safe.
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            gathered: list[dict | None] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, shards)
            merged: dict = {}
            for fragment in gathered:
                merged.update(fragment or {})
            shards = merged
        return {
            "stream_data": shards,
            "stream_world_size": self._world_size,
            "stream_source": source_fingerprint_from_config(self.config.data),
        }

    def load_checkpoint_state(self, state: dict | None) -> None:
        if not state:
            return
        validate_source_fingerprint(
            state.get("stream_source"),
            source_fingerprint_from_config(self.config.data),
        )
        saved_world_size = state.get("stream_world_size")
        if self._world_size is None:
            # Loaded before ``train_dataloader``: stash for deferred validation
            # (RIL ISS-204 — the pre-fix guard only fired when ``_world_size``
            # was already set, silently skipping the check for early loads).
            self._pending_world_size = None if saved_world_size is None else int(saved_world_size)
        else:
            self._validate_world_size(saved_world_size)
        self.stream_data_state = StreamDataState.from_dict(state.get("stream_data"))
        if self.train_dataset is not None:
            self.train_dataset.stream_data_state = self.stream_data_state
