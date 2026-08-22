"""Pipeline parallelism (PP) training helpers.

``parallel_strategy="pp"`` partitions a :class:`~llm.models.DecoderModel`
across ``pp_size`` pipeline stages — one stage per rank — and drives the
training / validation schedule with
:class:`torch.distributed.pipelining.ScheduleGPipe` (RIL DEC-049 / TASK-210).

v1 layout is a **pure pipeline**: the whole world is the pipeline
(``pp_size == world_size``, no data-parallel dimension). Every stage rank
loads the *same* data shard (mirroring pure-TP v1), and the schedule feeds
each batch into stage 0 and flows activations forward / gradients backward
stage by stage. Only the LAST stage computes the loss (LM shift + CE, see
:func:`lm_shift_loss`); the engine broadcasts it so every rank holds the same
value for metric reduction and ``save_best``.

The model is split at ``transformer_blocks``:

* stage 0        = embedding layer + the first block chunk
* stages 1..N-2  = middle block chunks
* stage N-1      = last block chunk + final norm + lm_head

Each stage's forward is semantics-preserving — it uses the exact block call
the monolithic :meth:`DecoderModel.forward` uses (``use_cache=False``,
``attn_mask=None``, causal by the block's own default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn

__all__ = [
    "allreduce_pp_dp_grads",
    "build_pipeline_model",
    "is_pp",
    "lm_shift_loss",
    "partition_decoder_model",
    "pp_dp_layout",
]


def pp_dp_layout(world_size: int, pp_size: int, rank: int | None = None) -> tuple[int, int, int, int]:
    """Resolve the PP + data-parallel 2D layout of the calling rank.

    PP+DP 2D (RIL TASK-211) arranges ranks in a row-major ``[DP][PP]`` grid:
    ``rank = dp_rank * pp_size + pp_rank``, so every pipeline group is a
    CONTIGUOUS run of ``pp_size`` ranks (the stage-to-stage P2P activation /
    gradient links stay within intranode-friendly blocks, mirroring the TP
    groups in ``tp_dp_layout``) and the ``dp_size`` DP groups are the strided
    columns — the ranks that hold the SAME stage across pipeline groups (the
    ones whose gradients must be averaged).

    Returns ``(pp_size, dp_size, dp_rank, pp_rank)``.

    Args:
        world_size: Total number of ranks.
        pp_size: Pipeline size; ``0`` or negative means "use the whole world
            as one pipeline group" (pure PP, no DP dimension).
        rank: Global rank (defaults to ``dist.get_rank()``; injectable for
            unit tests outside a live process group).

    Raises:
        ValueError: if ``pp_size`` exceeds ``world_size`` or does not divide
            it evenly (every rank must agree on the same grid).
    """
    if pp_size is None or pp_size <= 0:
        pp_size = world_size
    if pp_size > world_size:
        raise ValueError(f"pp_size={pp_size} cannot exceed world_size={world_size}.")
    if world_size % pp_size != 0:
        raise ValueError(
            f"pp_size={pp_size} must divide world_size={world_size} evenly for pipeline "
            "parallelism (each PP group is a contiguous world_size/pp_size-rank range)."
        )
    dp_size = world_size // pp_size
    if rank is None:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    dp_rank, pp_rank = divmod(rank, pp_size)
    return pp_size, dp_size, dp_rank, pp_rank


def lm_shift_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    scale: float = 1.0,
) -> torch.Tensor:
    """Causal-LM loss: ``logits[s]`` predicts ``labels[s + 1]``.

    Mirrors :meth:`llm.training.tasks.lm_task.LMTask.train_step` so a
    pipeline's last stage computes exactly the serial training loss
    (shift + ``CrossEntropyLoss(ignore_index=-100)``). ``scale`` carries the
    engine's gradient-accumulation factor (the schedule has already
    backpropagated by the time ``step()`` returns, so the 1/accum_steps
    scaling the standard loop applies to the loss must instead be folded
    into the schedule's own loss).
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    if shift_labels.numel() == 0:
        raise ValueError("sequence length must be > 1 for next-token language modeling")
    return criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) / scale


class _PipelineStage(nn.Module):
    """One sequential chunk of a DecoderModel, with a bound ``forward``.

    ``blocks`` holds the stage's slice of ``transformer_blocks`` (assigned by
    :func:`partition_decoder_model`); ``embedding_layer`` (stage 0),
    ``final_norm`` and ``lm_head`` (last stage) are attached as ``None`` when
    not owned so the same forward body handles every stage. ``forward`` takes
    ``input_ids`` on stage 0 and hidden states elsewhere, and returns hidden
    states — or ``logits`` on the last stage.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embedding_layer: nn.Module | None = None
        self.final_norm: nn.Module | None = None
        self.lm_head: nn.Module | None = None
        self.blocks: nn.ModuleList = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding_layer(x, start_pos=0, position_ids=None) if self.embedding_layer is not None else x
        for _i, blk in enumerate(self.blocks):
            # The exact training call DecoderModel.forward makes with
            # use_cache=False: causal by the block's own is_causal.
            h = blk(h, attn_mask=None, is_causal=None, use_cache=False, start_pos=0, layer_idx=None)
        if self.final_norm is not None:
            h = self.final_norm(h)
        if self.lm_head is not None:
            h = self.lm_head(h)
        return h


def partition_decoder_model(model: nn.Module, num_stages: int) -> list[nn.Module]:
    """Split a DecoderModel-family model into ``num_stages`` sequential stage modules.

    The split happens at ``transformer_blocks`` (stage 0 embeds + owns the
    first chunk; the last stage owns the final norm and lm_head), so each
    stage's forward takes a hidden-state tensor and returns one — except
    stage 0 (takes ``input_ids``) and the last stage (returns ``logits``).
    The stage modules *reference* the original model's submodules (same
    parameter objects), so distributed initialisation via
    ``broadcast_parameters`` on the full model works unchanged.

    Raises:
        NotImplementedError: for models without ``transformer_blocks``, with
            ``use_alibi``, with ``attn_impl='flash_attn'``, or with gradient
            checkpointing enabled — PP v1 does not cover these (kept out; the
            bit-exact parity against a serial reference is verified at the
            dropout=0 test configuration — with dropout > 0 each stage draws
            its own RNG stream and the parity is approximate, which is the
            standard GPipe caveat).
    """
    am = cast(Any, model)  # nn.Module attr access types as Tensor|Module; we know what we built
    if not (isinstance(getattr(am, "transformer_blocks", None), nn.ModuleList) and len(am.transformer_blocks) > 0):
        raise NotImplementedError(
            "Pipeline parallelism v1 partitions DecoderModel-family models that expose "
            f"a non-empty 'transformer_blocks' ModuleList, got {type(model).__name__}."
        )
    if getattr(am, "alibi", None) is not None:
        raise NotImplementedError(
            "Pipeline parallelism v1 does not partition ALiBi models (the shared bias "
            "module would be duplicated across stage state dicts)."
        )
    if getattr(am, "gradient_checkpointing", False):
        raise NotImplementedError(
            "Pipeline parallelism v1 does not support gradient checkpointing (the "
            "checkpointed block call differs from the plain call the stage uses)."
        )
    if getattr(am, "attn_impl", None) == "flash_attn":
        # PP v1 is fp32-only (the engine refuses use_amp), and the flash_attn
        # kernel rejects fp32 — running it here would crash on the first stage
        # forward with an opaque torch backend error instead of this setup
        # refusal (review MEDIUM on DEC-049/TASK-210).
        raise NotImplementedError(
            "Pipeline parallelism v1 does not support attn_impl='flash_attn' (the flash kernel "
            "requires half precision, but PP v1 is fp32-only). Use attn_impl='mha', 'sdpa' or "
            "'mla' for now."
        )

    blocks = am.transformer_blocks
    num_layers = len(blocks)
    boundaries = [num_layers * s // num_stages for s in range(num_stages + 1)]

    stages: list[nn.Module] = []
    for s in range(num_stages):
        lo, hi = boundaries[s], boundaries[s + 1]
        stage = _PipelineStage()
        if s == 0:
            stage.embedding_layer = am.embedding_layer
        stage.blocks = blocks[lo:hi]  # fresh ModuleList; blocks keep their own refs
        if s == num_stages - 1:
            stage.final_norm = am.final_norm
            stage.lm_head = am.lm_head
        stages.append(stage)

    return stages


def _build_name_maps(stage_module: nn.Module, lo: int) -> dict[str, str]:
    """Map a stage's state-dict keys onto the original model's global names.

    Stage-local keys are ``embedding_layer.*`` / ``final_norm.*`` /
    ``lm_head.*`` verbatim, and ``blocks.{j}.*`` which become
    ``transformer_blocks.{lo + j}.*``.
    """
    mapping: dict[str, str] = {}
    for key in stage_module.state_dict():
        if key.startswith("blocks."):
            rest = key[len("blocks.") :]
            if "." not in rest:
                raise RuntimeError(
                    f"unexpected stage state-dict key {key!r}: a block key must carry a "
                    "parameter/buffer suffix, not name the block module itself."
                )
            idx_str, suffix = rest.split(".", 1)
            mapping[key] = f"transformer_blocks.{lo + int(idx_str)}.{suffix}"
        else:
            mapping[key] = key
    return mapping


@dataclass
class _PPRuntime:
    """Per-rank PP metadata attached to the stage module as ``model._pp``."""

    stage_index: int
    num_stages: int
    group: dist.ProcessGroup | None  # None = the world group
    schedule: Any
    stage_module: nn.Module
    local_to_global: dict[str, str] = field(default_factory=dict)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dp_group: dist.ProcessGroup | None = None  # PP+DP 2D (TASK-211): strided DP column

    @property
    def last_stage_rank(self) -> int:
        return self.num_stages - 1

    def broadcast_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Broadcast a loss scalar from the last stage to every stage rank.

        Only the last stage computes the loss (``step``/``eval`` return it
        there); every other rank holds a detached zero. The engine needs the
        same value on every rank so ``reduce_mean`` (divisor = world) and
        ``save_best`` see the true pipeline loss.

        ``group_src`` (not ``src``): torch 2.13's ``broadcast`` treats ``src``
        as the source rank in the GLOBAL process group regardless of the
        ``group`` argument, and requires ``group_src`` for a group-LOCAL
        source. Under a PP subgroup (2D, TASK-211) ``self.last_stage_rank`` is
        the group-local index — passing it as ``src`` is only correct when the
        subgroup happens to contain that global rank (the world group / PP
        group 0 in a 2x2 grid) and deadlocks otherwise (the TASK-211 probe
        caught the group {2,3} case spinning forever).
        """
        t = loss.detach().to(self.device)
        dist.broadcast(t, group_src=self.last_stage_rank, group=self.group)
        return t

    def full_state_dict(self) -> dict[str, Any]:
        """Cross-stage collect: every stage's owned params into one global-named dict.

        Each rank gathers its own slice (global names via ``local_to_global``)
        and ``all_gather_object`` over the PP group, then every rank merges
        the union — a collective exactly like the TP full-state-dict gather,
        so the checkpoint manager's "every shard must enter it" contract
        (RIL ISS-186) applies.
        """
        mine = {self.local_to_global[k]: v for k, v in self.stage_module.state_dict().items()}
        gathered: list[dict[str, Any]] = [{} for _ in range(self.num_stages)]
        dist.all_gather_object(gathered, mine, group=self.group)
        merged: dict[str, Any] = {}
        for d in gathered:
            for name, value in d.items():
                # Stages own DISJOINT global-name slices; a collision means a
                # stage emitted a non-owned (e.g. duplicated) name and one
                # copy would be silently dropped from the checkpoint — fail
                # loud instead (review LOW on DEC-049/TASK-210).
                if name in merged:
                    raise RuntimeError(
                        f"pipeline stage overlap: global name {name!r} was produced by more than one "
                        "stage; the model is not partitioned into disjoint slices."
                    )
                merged[name] = value
        return merged

    def scatter_load(self, state_dict: dict[str, Any]) -> None:
        """Load the global-named full state dict, keeping only this stage's slice."""
        sub = {local: state_dict[global_name] for local, global_name in self.local_to_global.items()}
        self.stage_module.load_state_dict(sub, strict=True)


def build_pipeline_model(
    model: nn.Module,
    num_stages: int,
    device: torch.device,
    *,
    pp_group: dist.ProcessGroup | None = None,
    dp_group: dist.ProcessGroup | None = None,
    pp_rank: int | None = None,
) -> nn.Module:
    """Partition ``model`` into this rank's pipeline stage and schedule.

    v1 maps stages one-to-one onto ranks: ``stage_index == dist.get_rank()``
    and ``num_stages == world_size`` (a pure pipeline, no PP+DP hybrid yet).
    PP + data-parallel 2D (RIL TASK-211) passes a ``pp_rank`` (the group-local
    index in the ``[DP][PP]`` grid), a ``pp_group`` for the pipeline P2P and a
    ``dp_group`` for step-boundary gradient averaging; stage ``pp_rank`` then
    belongs to DP column ``dp_rank`` instead of being a whole-world stage.

    Weight initialisation is handled by the caller (the engine broadcasts
    rank 0's full model before wrapping); the stage modules *share* the
    model's parameter objects, so the broadcast reaches them.

    Returns the per-rank stage module tagged with ``_pp`` metadata
    (:func:`is_pp`), ready to be driven by ``runtime.step`` /
    ``runtime.eval`` from the engine training loop.
    """
    if num_stages <= 1:
        raise ValueError("pipeline parallelism needs num_stages > 1")
    if pp_rank is None:
        pp_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    partitions = partition_decoder_model(model, num_stages)
    stage_module = partitions[pp_rank].to(device)

    from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

    # Dynamic metadata mode: pass no ``input_args`` example so the stage's
    # recv/send buffers are inferred from the first real microbatch — batch
    # sequence lengths are dataset-dependent and a static example would
    # over-constrain them (the metadata validation rejects shape drift).
    # ``group=pp_group`` routes the P2P links over the pipeline subgroup
    # (TASK-211); ``None`` = the default/world group (pure PP v1).
    pipeline_stage = PipelineStage(stage_module, pp_rank, num_stages, device, input_args=None, group=pp_group)
    schedule = ScheduleGPipe(pipeline_stage, n_microbatches=1, loss_fn=lm_shift_loss)

    group = (
        pp_group
        if pp_group is not None
        else (dist.group.WORLD if dist.is_available() and dist.is_initialized() else None)
    )  # type: ignore[attr-defined]
    runtime = _PPRuntime(
        stage_index=pp_rank,
        num_stages=num_stages,
        group=group,
        schedule=schedule,
        stage_module=stage_module,
        device=device,
        dp_group=dp_group,
    )
    am = cast(Any, model)
    boundaries = [len(am.transformer_blocks) * s // num_stages for s in range(num_stages + 1)]
    runtime.local_to_global = _build_name_maps(stage_module, boundaries[pp_rank])

    cast(Any, stage_module)._pp = runtime
    return stage_module


def allreduce_pp_dp_grads(model: nn.Module) -> None:
    """Average a pipeline stage's gradients over its data-parallel group.

    PP + data-parallel 2D (RIL TASK-211): the strided DP group holds the SAME
    stage (same ``pp_rank``) on DIFFERENT data shards, so each copy's
    gradient only carries its own shard. Averaging over the DP group
    converges every stage to the true full-batch gradient (DDP semantics) —
    without it the identical stage copies across pipeline groups diverge and
    the model de-synchronises (the failure mode the bit-exact state-dict
    check in the 2D engine e2e guards).

    Runs AFTER unscale_ + partial-window re-scaling (fp32 reduce, correct
    scale) and ONLY at the step boundary — interior gradient-accumulation
    microbatches must not sync (the same rule the DDP ``no_sync`` branch
    enforces). Each stage owns DISJOINT parameters, so every rank's ``param``
    list is exactly its stage slice; the average is uniform because all
    copies have the same stage structure.

    A no-op for pure PP (``dp_group`` is ``None``) and non-PP models.
    """
    if not is_pp(model):
        return
    dp_group = getattr(cast(Any, model)._pp, "dp_group", None)
    if dp_group is None:
        return
    for param in model.parameters():
        if param.grad is None:
            # Keep the collective sequence uniform across the DP group: a peer
            # whose stage param got a gradient enters the reduce, so a skipped
            # call would deadlock. A stage param with no grad contributes
            # exactly ZERO (find_unused_parameters semantics).
            param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=dp_group)


def is_pp(model: nn.Module) -> bool:
    """True when ``model`` is a pipeline stage (tagged by ``build_pipeline_model``).

    Makes checkpoint save/load a cross-stage collective, like
    :func:`llm.training.distributed.parallel.is_tp` for the TP identity.
    """
    return getattr(model, "_pp", None) is not None
