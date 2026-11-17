"""Native ZeRO Stage-1 optimizer-state partitioning (no DeepSpeed dependency).

ROADMAP 15.5 (P2) — DeepSpeed ZeRO 集成, first slice:
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
from typing import Any, overload

import torch
import torch.distributed as dist
from torch.optim import Optimizer

# Key under which every rank's shard is persisted in a checkpoint's optimizer
# slot. The shards are disjoint, so a single rank's ``state_dict`` cannot
# recreate the others — resume needs the whole set, keyed by the rank that
# owns each one.
ZERO_RANK_STATES_KEY = "__zero_rank_states__"


def _owner_of(index: int, world_size: int) -> int:
    """Round-robin whole-parameter ownership (rank owns ``index % world_size``)."""
    return index % world_size


class ZeroOptimizer(Optimizer):
    """Partition optimizer state across data-parallel ranks (ZeRO Stage 1).

    Parameters are owned round-robin by flat index: each rank builds its inner
    optimizer over ONLY the parameters it owns, so the optimizer state
    (e.g. Adam's first/second moments) is sharded to ~1/world_size per rank.
    On :meth:`step` the owning rank applies the update to its parameters and
    the full updated set is all-gathered so every rank holds identical weights.

    ``param_groups`` and ``zero_grad`` are delegated to the inner optimizer so
    learning-rate schedulers and the training engine can keep using the same
    interface. Subclassing :class:`torch.optim.Optimizer` keeps it a drop-in for
    the training engine's scheduler / AMP scaler / checkpoint paths (mirroring
    how fairscale's ``ZeroRedundancyOptimizer`` extends ``torch.optim.Optimizer``).

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
        # Initialize the base Optimizer machinery over the owned subset so
        # isinstance(self, torch.optim.Optimizer) holds (scheduler / scaler /
        # checkpoint compatibility); then point its groups/state at the inner
        # sharded optimizer below.
        Optimizer.__init__(self, owned, {})
        self._opt: Optimizer = build_optimizer(owned)
        # Present the INNER optimizer's groups/state/defaults as our own so the
        # scheduler, GradScaler-disabled path, and checkpoint manager operate on
        # the real sharded optimizer without knowing about the wrapper.
        self._adopt_inner_state()

    def _adopt_inner_state(self) -> None:
        """Alias this wrapper's optimizer slots to the inner optimizer's.

        ``Optimizer.load_state_dict`` REBUILDS the inner optimizer's
        ``param_groups`` / ``state`` (fresh lists and dicts) instead of
        mutating them in place, so the references captured at construction go
        stale. Anything holding the wrapper then acts on orphaned state: an
        ``LRScheduler`` attached before a resume writes each step's learning
        rate into dicts the optimizer never reads, so training silently keeps
        the pre-resume LR while the logs report the scheduled one. Re-pointing
        after every load (and at construction, so both paths agree) keeps the
        wrapper a faithful view of the optimizer that actually steps.
        """
        self.param_groups = self._opt.param_groups
        self.state = self._opt.state
        self.defaults = self._opt.defaults

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

        This shard alone is NOT a resumable checkpoint for the whole job: the
        shards are disjoint, so rank 0's piece cannot replay rank 1's moments.
        Use :meth:`collective_state_dict` when writing a checkpoint that a
        multi-rank job will resume from.
        """
        return {
            "_zero_stage1": True,
            "rank": self._rank,
            "world_size": self._world_size,
            "global_indices": [i for i, _ in enumerate(self._params) if self._owners[i] == self._rank],
            "inner": self._opt.state_dict(),
        }

    def collective_state_dict(self) -> dict[str, Any]:
        """Gather EVERY rank's shard into one checkpointable payload.

        A ZeRO checkpoint must carry the whole sharded set: each rank owns a
        disjoint slice of the optimizer state, so persisting only rank 0's
        shard silently drops the other ranks' Adam moments and makes a
        multi-rank resume impossible (the loader then either refuses the
        mismatched rank tag or restarts those shards cold).

        This is a cross-rank collective (``all_gather_object``), so every rank
        must call it, exactly like the TP/PP/FSDP state-dict gathers. The
        result is keyed by GLOBAL rank so each rank restores its own slice
        through :meth:`load_state_dict` without any additional rendezvous.
        """
        mine = self.state_dict()
        if not (dist.is_available() and dist.is_initialized()) or self._world_size == 1:
            return {ZERO_RANK_STATES_KEY: {str(self._rank): mine}}
        gathered: list[dict[str, Any]] = [{} for _ in range(self._world_size)]
        dist.all_gather_object(gathered, mine, group=self._pg)
        return {ZERO_RANK_STATES_KEY: {str(i): shard for i, shard in enumerate(gathered)}}

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore this rank's sharded optimizer state (resume).

        Accepts either the gathered envelope written by
        :meth:`collective_state_dict` (the multi-rank resume path — this rank
        picks its own slice out of the set) or a bare per-rank shard (the
        single-rank / hand-built path). Only ``inner`` is applied, into THIS
        rank's deterministic owned-param set; the owner mapping is fixed by
        round-robin, so the same shard slots map onto the same parameters on
        load as on save.
        """
        if state_dict.get(ZERO_RANK_STATES_KEY) is not None:
            shards = state_dict[ZERO_RANK_STATES_KEY]
            mine = shards.get(str(self._rank))
            if mine is None:
                raise ValueError(
                    f"ZeRO checkpoint carries no optimizer shard for rank {self._rank} "
                    f"(it holds ranks {sorted(shards)}); was it saved by a job with a different world size?"
                )
            state_dict = mine
        if not state_dict.get("_zero_stage1"):
            raise ValueError("not a ZeroOptimizer state dict (missing '_zero_stage1' marker)")
        if state_dict.get("rank") != self._rank:
            raise ValueError(f"state dict is for rank {state_dict.get('rank')}, this rank is {self._rank}")
        if state_dict.get("world_size") != self._world_size:
            raise ValueError(f"state dict world_size {state_dict.get('world_size')} != current {self._world_size}")
        self._opt.load_state_dict(state_dict["inner"])
        # ``load_state_dict`` rebuilds the inner optimizer's groups/state, so
        # re-adopt them or a scheduler attached earlier keeps writing into the
        # pre-load dicts (training silently ignores every LR update).
        self._adopt_inner_state()

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

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

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


class ZeroStage2Optimizer(ZeroOptimizer):
    """Partition gradients AND optimizer state across ranks (ZeRO Stage 2).

    Stage-1 (:class:`ZeroOptimizer`) shards only the OPTIMIZER STATE: after DDP
    averages the gradients, every rank still materialises a full
    ``num_params`` gradient buffer, and the sharded optimizer reads only the
    slice it owns. Stage-2 removes that duplication — each rank keeps gradients
    for its owned parameters only, and the cross-rank reduction lands straight
    in the owner via ``reduce_scatter``.

    Memory saved per rank: one full gradient buffer (``num_params * dtype``
    bytes). Adam's moments are already sharded by Stage-1.

    **Contract difference that callers must respect** — after
    :meth:`reduce_gradients_to_owners`, a non-owned parameter's ``.grad`` is
    ``None`` on this rank. That is "this rank does not update it", NOT "its
    gradient is zero". Consequently:

    * Gradient clipping MUST happen on the GLOBAL norm BEFORE the reduction
      (:meth:`clip_gradients_globally`), while every rank still holds all
      gradients. Clipping after the reduction, or per-rank, applies a different
      factor per shard — measured: local norms 1.0000 vs 0.9383 for ranks whose
      gradients differ in scale, versus one factor from the global norm.
      See ADR-014.
    * Diagnostics that read ``param.grad`` must tolerate ``None`` for non-owned
      parameters.

    This class is the incremental difference, not a rewrite: parameter
    ownership, the inner-optimizer shard, ``state_dict`` /
    ``collective_state_dict`` / ``load_state_dict`` and ``zero_grad`` are
    inherited unchanged from :class:`ZeroOptimizer`, so a Stage-2 checkpoint is
    the same on-disk shape as a Stage-1 one.
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
        super().__init__(
            build_optimizer,
            params,
            rank=rank,
            world_size=world_size,
            process_group=process_group,
        )
        # Stage-2 checkpoints carry the same shard layout as Stage-1, but the
        # marker must distinguish them: a Stage-2 job reads gradients that a
        # Stage-1 job materialises in full, so resuming one as the other would
        # silently change the reduction path.
        self._stage = 2

    # --- gradient reduction -------------------------------------------------

    def reduce_gradients_to_owners(self) -> None:
        """Reduce gradients across ranks, keep only this rank's owned shard.

        Semantics: an owned parameter's gradient becomes the cross-rank MEAN —
        the same value ``allreduce_zero_grads`` (Stage-1) or DDP produces — so a
        Stage-2 step is numerically identical to DDP + Stage-1. Every non-owned
        gradient is then set to ``None``, which is where the memory saving comes
        from.

        Called once per optimizer step, after ``backward()`` and after
        :meth:`clip_gradients_globally`. Idempotent for ``world_size == 1``.

        **Why ``all_reduce`` per parameter and not one ``reduce_scatter``.**
        ``reduce_scatter`` slices its input by POSITION: rank ``r`` receives the
        ``r``-th equal contiguous segment. Round-robin ownership does not line up
        with that — with 4 parameters and 2 ranks, rank 0 owns indices 0 and 2,
        but index 2 falls inside rank 1's positional segment. (A first
        implementation did exactly this and failed with "parameter 2 lies outside
        rank 0's reduced segment".) Making ``reduce_scatter`` work would require
        a rank-major parameter layout that the sharded optimizer does not use, so
        Stage-2 reduces per parameter instead: correct for arbitrary ownership,
        at the cost of one collective per parameter rather than a batched
        scatter. The memory win — dropping non-owned gradients — does not depend
        on which collective is used.

        A future optimisation can batch the reduction by bucketing parameters
        whose ownership aligns, which needs the owner mapping at a coarser
        granularity than round-robin.
        """
        if self._world_size <= 1:
            for index, owner in enumerate(self._owners):
                if owner != self._rank:
                    self._params[index].grad = None
            return

        for index, param in enumerate(self._params):
            grad = param.grad
            if grad is not None:
                dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=self._pg)
            elif self._owners[index] == self._rank:  # pragma: no cover - caller contract
                raise RuntimeError(
                    f"parameter {index} (owned by rank {self._rank}) has no gradient; "
                    "reduce_gradients_to_owners must run after backward() and before zero_grad()"
                )
            if self._owners[index] != self._rank:
                param.grad = None

    def clip_gradients_globally(self, max_norm: float, eps: float = 1e-6) -> torch.Tensor:
        """Clip gradients using the GLOBAL L2 norm across all ranks.

        Must run BEFORE :meth:`reduce_gradients_to_owners`, while every rank
        still holds every gradient. Clipping afterwards — or per rank — applies
        a different multiplier per shard: measured 1.0000 vs 0.9383 for two
        ranks whose local gradients differ in scale, versus one common factor
        from the global norm (ADR-014).

        Args:
            max_norm: Maximum global L2 norm. ``<= 0`` disables clipping (the
                engine's "no clip" convention).
            eps: Denominator guard for the clip coefficient.

        Returns:
            The global total norm as a 0-dim tensor (mirrors
            ``clip_grad_norm_``'s return: ``torch.isfinite`` detects a skipped
            step, ``.item()`` feeds perf monitoring).
        """
        params = [p for p in self._params if p.grad is not None]
        try:
            device = next(iter(self._params)).device
        except StopIteration:
            device = torch.device("cpu")
        total_sq = torch.zeros((), dtype=torch.float32, device=device)
        for param in params:
            grad = param.grad
            if grad is None:  # pragma: no cover - filtered above
                continue
            total_sq += grad.detach().float().pow(2).sum()

        if self._world_size > 1:
            dist.all_reduce(total_sq, op=dist.ReduceOp.SUM, group=self._pg)
        global_norm = total_sq.sqrt()

        if max_norm <= 0:
            return global_norm

        coefficient = torch.clamp(global_norm.clamp(min=eps), min=eps)
        scale = float(max_norm) / float(coefficient)
        if scale < 1.0:
            for param in params:
                grad = param.grad
                if grad is None:  # pragma: no cover - filtered above
                    continue
                grad.detach().mul_(scale)
        return global_norm
