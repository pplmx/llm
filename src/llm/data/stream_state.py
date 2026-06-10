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
        shard_id = rank * num_workers + worker_id
        return str(shard_id)

    def get_shard(self, rank: int, worker_id: int, num_workers: int) -> StreamShardState:
        key = self.shard_key(rank, worker_id, num_workers)
        if key not in self.shards:
            self.shards[key] = StreamShardState()
        return self.shards[key]

    def to_dict(self) -> dict:
        return {key: shard.to_dict() for key, shard in self.shards.items()}

    @classmethod
    def from_dict(cls, data: dict | None) -> StreamDataState:
        if not data:
            return cls()
        return cls(shards={key: StreamShardState.from_dict(value) for key, value in data.items()})
