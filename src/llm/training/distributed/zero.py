"""Native ZeRO Stage-1 optimizer-state partitioning (no DeepSpeed dependency).

ROADMAP 15.5 (P2) — DeepSpeed ZeRO 集成, first slice (RIL TASK-267/ISS-303):
a self-contained :class:`ZeroOptimizer` that partitions optimizer state across
data-parallel ranks so per-rank optimizer-state memory is ~1/world_size of the
full set, then all-gathers the updated parameters so every rank converges to
identical weights after each step.

ZeRO Stage-1 assumption (shared with upstream DeepSpeed): gradients are already
averaged across the data-parallel group before ``step`` (e.g. by DDP). Under
that condition the owner of each parameter applies exactly the same
globally-correct update the unsharded optimizer would, and the parameter
all-gather restores a full, identical weight copy on every rank.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
import torch.distributed as dist
from torch.optim import Optimizer


def _owner_of(index: int, world_size: int) -> int:
    """Round-robin whole-parameter ownership (rank owns ``index % world_size``)."""
    return index % world_size


class ZeroOptimizer:
    """Partition optimizer state across data-parallel ranks (ZeRO Stage 1).

    Parameters are owned round-robin by flat index: each rank builds its inner
    optimizer over ONLY the parameters it owns, so the optimizer state
    (e.g. Adam's first/second moments) is sharded to ~1/world_size per rank.
    On :meth:`step` the owning rank applies the update to its parameters and
    the full updated set is all-gathered so every rank holds identical weights.

    ``param_groups`` and ``zero_grad`` are delegated to the inner optimizer so
    learning-rate schedulers and the training engine can keep using the same
    interface.

    Args:
        build_optimizer: callable ``(owned_params) -> Optimizer`` building the
            inner optimizer over a *subset* of the full parameter list.
        params: the full (identical on every rank) parameter list.
        rank / world_size: current rank and group size. Auto-detected from the
            process group when omitted.
        process_group: data-parallel process group (``None`` = default group).
    """

    def __init__(
        self,
        build_optimizer: Callable[[list[torch.nn.Parameter]], Optimizer],
        params: Iterable[torch.nn.Parameter],
        *,
        rank: int | None = None,
        world_size: int | None = None,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        params = list(params)
        if world_size is None:
            world_size = _world_size(process_group)
        if rank is None:
            rank = _rank(process_group)
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank {rank} out of range for world_size {world_size}")
        self._pg = process_group
        self._rank = rank
        self._world_size = world_size
        self._params = params
        self._owners = [_owner_of(i, world_size) for i in range(len(params))]
        owned = [p for i, p in enumerate(params) if self._owners[i] == rank]
        if not owned:
            raise ValueError(f"rank {rank} owns no parameters (world_size {world_size})")
        self._opt: Optimizer = build_optimizer(owned)
        self.param_groups = self._opt.param_groups

    @property
    def optimizer(self) -> Optimizer:
        """The inner (sharded) optimizer built over this rank's owned params."""
        return self._opt

    def owned_count(self) -> int:
        """Number of parameters this rank owns (its shard width)."""
        return len(self._opt.param_groups[0]["params"]) if self._opt.param_groups else 0

    def state_dict(self) -> dict:
        """Return this rank's sharded optimizer state for checkpoint / resume.

        The inner optimizer only holds this rank's owned parameters, so its own
        ``state_dict`` is already the correct per-rank shard (FSDP-sharded
        style) and needs no collective to serialize. ``global_indices`` records
        which full-model parameter positions this shard covers so a loader can
        audit the mapping or rebuild a gathered full state if ever needed.
        """
        return {
            "_zero_stage1": True,
            "rank": self._rank,
            "world_size": self._world_size,
            "global_indices": [i for i, _ in enumerate(self._params) if self._owners[i] == self._rank],
            "inner": self._opt.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore this rank's sharded optimizer state (resume).

        Only ``inner`` is applied, into THIS rank's deterministic owned-param
        set; the owner mapping is fixed by round-robin, so the same shard slots
        map onto the same parameters on load as on save.
        """
        if not state.get("_zero_stage1"):
            raise ValueError("not a ZeroOptimizer state dict (missing '_zero_stage1' marker)")
        if state.get("rank") != self._rank:
            raise ValueError(f"state dict is for rank {state.get('rank')}, this rank is {self._rank}")
        if state.get("world_size") != self._world_size:
            raise ValueError(f"state dict world_size {state.get('world_size')} != current {self._world_size}")
        self._opt.load_state_dict(state["inner"])

    def zero_grad(self, set_to_none: bool = True) -> None:
        # Zero every parameter's gradient, not just this rank's owned shard:
        # the engine / DDP expectation is that a full zero happens each step,
        # and leaving non-owned grads to accumulate would silently grow memory
        # on every rank while never being consumed by the local optimizer.
        for param in self._params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.zero_()
        self._opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        result = self._opt.step(closure)
        if self._world_size > 1:
            self._sync()
        return result

    def _sync(self) -> None:
        """All-gather each parameter's full value from its owner.

        The owner's copy was updated in place by ``self._opt.step``; every other
        rank holds the pre-step (stale) copy for that parameter. Gathering all
        ``world_size`` copies and picking the owner's restores identical,
        fully-updated weights on every rank.
        """
        for index, param in enumerate(self._params):
            owner = self._owners[index]
            value = param.detach()
            gathered = [torch.empty_like(value) for _ in range(self._world_size)]
            dist.all_gather(gathered, value, group=self._pg)
            with torch.no_grad():
                param.copy_(gathered[owner])


def _world_size(process_group: dist.ProcessGroup | None) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(process_group)
    return 1


def _rank(process_group: dist.ProcessGroup | None) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(process_group)
    return 0
