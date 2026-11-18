"""Checkpointable state for streaming IterableDataset shards."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StreamShardState:
    """Resume cursor for one DDP rank x DataLoader worker shard."""

    line_index: int = 0
    token_buffer: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"line_index": self.line_index, "token_buffer": self.token_buffer}

    @classmethod
    def from_dict(cls, data: dict | None) -> StreamShardState:
        if not data:
            return cls()
        return cls(
            line_index=int(data.get("line_index", 0)),
            token_buffer=list(data.get("token_buffer", [])),
        )


@dataclass
class StreamDataState:
    """Collection of per-shard streaming cursors."""

    shards: dict[str, StreamShardState] = field(default_factory=dict)

    @staticmethod
    def shard_key(rank: int, worker_id: int, num_workers: int) -> str:
        """Key for one (rank, worker) cursor.

        ``rank * num_workers + worker_id`` only separates ranks when
        ``num_workers >= 1``: with 0 every rank collapses to key ``"0"`` and all
        ranks would SHARE one resume cursor, so a resumed run silently skips or
        replays data on every rank but the first. Measured: 4 ranks with
        ``num_workers=0`` produced a single key.

        The streaming dataset always passes at least 1 (it substitutes 1 for the
        single-process loader), so this is a guard against a future caller rather
        than a live bug — but a silent shared cursor is the kind of failure that
        only shows up as slightly-wrong training data, so it is refused here.
        """
        if num_workers < 1:
            raise ValueError(
                f"num_workers must be >= 1 to compute a shard key, got {num_workers}; "
                "0 would map every rank to the same cursor"
            )
        if not 0 <= worker_id < num_workers:
            raise ValueError(f"worker_id must be in [0, {num_workers}), got {worker_id}")
        shard_id = rank * num_workers + worker_id
        return str(shard_id)

    def get_shard(self, rank: int, worker_id: int, num_workers: int) -> StreamShardState:
        key = self.shard_key(rank, worker_id, num_workers)
        if key not in self.shards:
            self.shards[key] = StreamShardState()
        return self.shards[key]

    def reset(self) -> None:
        """Zero all per-shard cursors.

        Used when a streaming corpus is exhausted before the step budget is
        met: the next iteration restarts the corpus from the beginning
        (streaming pretraining cycles the corpus until ``steps_per_epoch``
        completes).
        """
        self.shards.clear()

    def to_dict(self) -> dict:
        return {key: shard.to_dict() for key, shard in self.shards.items()}

    @classmethod
    def from_dict(cls, data: dict | None) -> StreamDataState:
        if not data:
            return cls()
        return cls(shards={key: StreamShardState.from_dict(value) for key, value in data.items()})
