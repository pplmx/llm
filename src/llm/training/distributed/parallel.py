"""Parallel training strategy helpers (DDP, FSDP).

The two strategies are exposed through a single entry point
(:func:`wrap_model_for_training`) so the trainer doesn't have to
branch on ``parallel_strategy``. DDP is the default; FSDP is
opt-in via ``DistributedConfig.parallel_strategy = "fsdp"``.

FSDP configuration (mixed precision, auto-wrap threshold, CPU
offload) is read from the same ``DistributedConfig`` so users
can tune it without forking the trainer.

Note:
    FSDP requires an initialised process group (``init_process_group``
    or ``torchrun``). These helpers run inside the trainer, which
    has already done the init by the time it calls
    :func:`wrap_model_for_training`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from llm.training.distributed.pipeline import build_pipeline_model, is_pp, pp_dp_layout
from llm.training.distributed.tensor_parallel import apply_tensor_parallel

# State-dict strategy for FSDP save / load.
#
# - ``"full"`` materialises the full state dict on rank 0. Easier
#   for single-host save/load and for portability (the result is
#   a regular ``state_dict``), but defeats the memory savings of
#   FSDP because rank 0 needs to hold the whole model briefly.
# - ``"sharded"`` keeps each rank's shard locally. Best for
#   large-scale save/load because memory stays bounded by the
#   shard size, but the resulting file is only readable by a
#   parallel run with the same world size.
StateDictType = Literal["full", "sharded"]


def tp_dp_layout(world_size: int, tp_size: int, rank: int | None = None) -> tuple[int, int, int, int]:
    """Resolve the TP + data-parallel 2D layout of the calling rank.

    TP+DP 2D (RIL TASK-202) arranges ranks in a row-major ``[DP][TP]`` grid:
    ``rank = dp_rank * tp_size + tp_rank``, so every TP group is a CONTIGUOUS
    run of ``tp_size`` ranks (intra-node friendly for the TP all-reduces) and
    the ``dp_size`` DP groups are strided — the ranks that hold the SAME model
    shard across TP groups (the ones whose gradients must be averaged).

    Returns ``(tp_size, dp_size, dp_rank, tp_rank)``.

    Args:
        world_size: Total number of ranks.
        tp_size: Tensor-parallel size; ``0`` or negative means "use the whole
            world as one TP group" (pure TP, no DP dimension).
        rank: Global rank (defaults to ``dist.get_rank()``; injectable for
            unit tests outside a live process group).

    Raises:
        ValueError: if ``tp_size`` exceeds ``world_size`` or does not divide
            it evenly (every rank must agree on the same grid).
    """
    if tp_size is None or tp_size <= 0:
        tp_size = world_size
    if tp_size > world_size:
        raise ValueError(f"tp_size={tp_size} cannot exceed world_size={world_size}.")
    if world_size % tp_size != 0:
        raise ValueError(
            f"tp_size={tp_size} must divide world_size={world_size} evenly for tensor "
            "parallelism (each TP group is a contiguous world_size/tp_size-rank range)."
        )
    dp_size = world_size // tp_size
    if rank is None:
        rank = dist.get_rank() if dist.is_available() else 0
    dp_rank, tp_rank = divmod(rank, tp_size)
    return tp_size, dp_size, dp_rank, tp_rank


def _fsdp_mixed_precision(dtype: str) -> Any | None:
    """Build a ``MixedPrecision`` policy from the ``fsdp_mixed_precision`` string.

    Returns ``None`` for ``"fp32"`` (FSDP defaults to FP32 in that case).
    """
    if dtype == "fp32":
        return None
    if dtype == "bf16":
        from torch.distributed.fsdp import MixedPrecision

        param_dtype = torch.bfloat16
        reduce_dtype = torch.bfloat16
        buffer_dtype = torch.bfloat16
        return MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )
    if dtype == "fp16":
        # Refuse rather than silently produce an un-winnable run (RIL ISS-188):
        # FP16 parameters/reductions need a loss scaler or gradients underflow
        # and the run never converges. The engine only wires a GradScaler for
        # float16 AMP (not the default bf16/auto path), and FSDP's own scaling
        # is not wired here. Fail fast; the user should pick bf16 (on capable
        # GPUs) or fp32.
        raise ValueError(
            "fsdp_mixed_precision='fp16' is not supported: FP16 sharded parameters and "
            "gradient reductions require a loss scaler the framework does not wire. "
            "Use 'bf16' (on GPUs that support it) or 'fp32'."
        )
    raise ValueError(f"Unsupported fsdp_mixed_precision {dtype!r}; expected 'fp32', 'bf16', or 'fp16'.")


def _fsdp_auto_wrap_policy(min_params: int) -> Any | None:
    """Build a size-based auto-wrap policy. Returns ``None`` if disabled (``min_params <= 0``)."""
    if min_params <= 0:
        return None
    from functools import partial

    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    # ``size_based_auto_wrap_policy`` is the policy function itself —
    # FSDP expects a callable taking ``(module, recurse, nonwrapped_numel)``.
    # ``functools.partial`` binds ``min_num_params`` so the resulting
    # policy has the right signature.
    return partial(size_based_auto_wrap_policy, min_num_params=min_params)


def _build_fsdp(
    model: nn.Module,
    device: torch.device,
    *,
    mixed_precision: str,
    auto_wrap_min_params: int,
    cpu_offload: bool,
) -> nn.Module:
    """Construct the FSDP wrapper with the configured policy / dtype / offload."""
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel

    mp_policy = _fsdp_mixed_precision(mixed_precision)
    auto_wrap = _fsdp_auto_wrap_policy(auto_wrap_min_params)
    offload = CPUOffload(offload_params=True) if cpu_offload else None

    kwargs: dict[str, Any] = {
        "device_id": device.index,
        "mixed_precision": mp_policy,
    }
    if auto_wrap is not None:
        kwargs["auto_wrap_policy"] = auto_wrap
    if offload is not None:
        kwargs["cpu_offload"] = offload
    return FullyShardedDataParallel(model, **kwargs)


def wrap_model_for_training(
    model: nn.Module,
    *,
    parallel_strategy: str,
    device: torch.device,
    world_size: int,
    fsdp_mixed_precision: str = "bf16",
    fsdp_auto_wrap_min_params: int = 10_000_000,
    fsdp_cpu_offload: bool = False,
    tp_size: int = 1,
    pp_size: int = 0,
    pp_n_microbatches: int = 1,
) -> nn.Module:
    """Wrap a model for distributed training.

    For single-rank or CPU runs the model is returned unchanged —
    wrapping a CPU model in ``DistributedDataParallel`` is a no-op
    that can confuse some optimisers, and ``FullyShardedDataParallel``
    requires CUDA.

    Args:
        model: The bare ``nn.Module`` to wrap.
        parallel_strategy: ``"ddp"``, ``"fsdp"``, ``"tp"`` or ``"pp"``
            (pipeline parallelism, RIL DEC-049/TASK-210 — see
            :mod:`llm.training.distributed.pipeline`).
        device: Target device (used to set ``device_ids`` for DDP
            and ``device_id`` for FSDP).
        world_size: Number of ranks in the process group.
        fsdp_mixed_precision: FSDP parameter / gradient dtype.
            One of ``"fp32"``, ``"bf16"``, ``"fp16"``.
        fsdp_auto_wrap_min_params: FSDP size-based auto-wrap
            threshold. ``0`` disables auto-wrap.
        fsdp_cpu_offload: Offload FSDP params to CPU when idle.
        tp_size: Tensor-parallel size for ``parallel_strategy="tp"``.
            v1 uses the whole world as one TP group, so it defaults to
            ``world_size``; must be > 1 and divide every partitioned axis.
        pp_size: Pipeline size for ``parallel_strategy="pp"``. ``0`` (default)
            means "use the whole world as one pipeline group" (pure PP v1,
            RIL DEC-049/TASK-210). A value below ``world_size`` enables PP +
            data-parallel 2D (RIL TASK-211): ranks are laid out row-major as
            ``[DP][PP]`` — each pipeline group is a CONTIGUOUS
            ``world_size/pp_size``-rank range whose stage-to-stage P2P stays
            intranode-friendly, and the ``world_size/pp_size`` DP groups (the
            strided columns holding the same stage) average gradients across
            data shards at each step. ``world_size`` must divide evenly by
            ``pp_size``.
        pp_n_microbatches: Pipeline microbatch count for
            ``parallel_strategy="pp"`` (RIL TASK-213). ``1`` (default) runs
            one chunk per batch; ``> 1`` chunks each batch so the schedule can
            overlap stages and shrink per-stage activation memory. The
            schedule normalizes the gradient by the microbatch count, so the
            optimizer step is numerically unchanged.

    Raises:
        ValueError: if ``parallel_strategy`` is not recognised.
        RuntimeError: if FSDP is requested on a CPU-only host
            (FSDP needs CUDA + a process group).
        NotImplementedError: for unsupported tensor-parallel
            configurations (see ``apply_tensor_parallel`` scope guards).
    """
    if parallel_strategy == "pp":
        # Pipeline parallelism can run on CPU ranks (gloo) for 2-process
        # verification of numerics, exactly like TP — so it does NOT take the
        # DDP/FSDP CPU-no-op early return. Single-rank PP stays a no-op
        # identity (a 1-stage pipeline is the serial model).
        if world_size <= 1:
            return model
        if is_pp(model) or is_tp(model):
            raise ValueError("pipeline parallelism cannot be composed with another parallel_strategy in v1")
        pp_size, dp_size, dp_rank, pp_rank = pp_dp_layout(world_size, pp_size)
        if dp_size > 1:
            # PP + data-parallel 2D (TASK-211): ``dist.new_group`` is a
            # collective over the WORLD group, so EVERY rank must create the
            # SAME set of subgroups in the SAME order; each rank then picks
            # the handles for the groups it belongs to (mirrors the TP+DP 2D
            # rule, RIL TASK-202). Pure PP (dp_size == 1) skips subgroup
            # creation entirely and keeps using the default group.
            pp_groups = [dist.new_group(ranks=list(range(d * pp_size, (d + 1) * pp_size))) for d in range(dp_size)]
            dp_groups = [dist.new_group(ranks=list(range(p, world_size, pp_size))) for p in range(pp_size)]
            pp_group = pp_groups[dp_rank]
            dp_group = dp_groups[pp_rank]
        else:
            pp_group, dp_group = None, None
        return build_pipeline_model(
            model,
            pp_size,
            device,
            pp_group=pp_group,
            dp_group=dp_group,
            pp_rank=pp_rank,
            n_microbatches=pp_n_microbatches,
        )

    if parallel_strategy != "tp" and (world_size <= 1 or device.type != "cuda"):
        # DDP/FSDP are no-ops for single-rank or CPU-only runs.
        return model

    if parallel_strategy == "ddp":
        # MoE models MUST opt into unused-parameter tracking (RIL ISS-138):
        # a batch that routes zero tokens to an expert leaves that expert
        # outside the autograd graph, and DDP (find_unused_parameters=False)
        # then fails the backward with "expected to have finished reduction
        # (but one or more parameters still have no gradient)". Dead experts
        # are the structural norm for MoE (top-k routing over many experts),
        # not an anomaly, so blanket-unused tracking is the correct trade —
        # and it stays OFF for standard models where every param is used
        # every step (a real reduction-path speedup).
        uses_moe = any(getattr(module, "num_experts", 0) > 0 for module in model.modules())
        return DistributedDataParallel(
            model,
            device_ids=[device.index],
            find_unused_parameters=uses_moe,
        )

    if parallel_strategy == "fsdp":
        return _build_fsdp(
            model,
            device,
            mixed_precision=fsdp_mixed_precision,
            auto_wrap_min_params=fsdp_auto_wrap_min_params,
            cpu_offload=fsdp_cpu_offload,
        )

    if parallel_strategy == "tp":
        # Tensor parallelism CAN run on CPU ranks (gloo) — the early
        # DDP/FSDP CPU no-op deliberately does not apply here so 2-process
        # CPU verification of TP numerics is possible. Single-rank TP stays
        # a no-op identity (mirrors DDP/FSDP world_size<=1).
        if world_size <= 1:
            return model
        tp_size, dp_size, dp_rank, tp_rank = tp_dp_layout(world_size, tp_size)
        # Row-major [DP][TP] grid (world_size % tp_size == 0 guaranteed by
        # ``tp_dp_layout``): TP groups are contiguous rank ranges, DP groups
        # are strided. ``dist.new_group`` is a collective over the WORLD
        # group, so EVERY rank must create the SAME set of subgroups in the
        # SAME order (building only "my own" groups would mismatch the other
        # ranks' calls and deadlock); each rank then picks the handles for
        # the groups it belongs to.
        tp_groups = [dist.new_group(ranks=list(range(d * tp_size, (d + 1) * tp_size))) for d in range(dp_size)]
        dp_groups = [dist.new_group(ranks=list(range(t, world_size, tp_size))) for t in range(tp_size)]
        tp_group = tp_groups[dp_rank]
        dp_group = dp_groups[tp_rank]
        model = apply_tensor_parallel(cast(Any, model), process_group=tp_group)
        if dp_size > 1:
            # TP + data-parallel 2D: record the DP group so the engine (or any
            # caller) can average gradients across data shards at step
            # boundaries via :func:`allreduce_dp_grads`. dp_size == 1 is pure
            # TP v1 (every rank already sees identical replicated data).
            cast(Any, model)._tp.dp_group = dp_group
        return model

    raise ValueError(f"Unknown parallel_strategy '{parallel_strategy}'. Expected 'ddp', 'fsdp', 'tp' or 'pp'.")


def _strip_compile_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize torch.compile state-dict keys to the plain module namespace.

    ``torch.compile(model).state_dict()`` prefixes every key with
    ``_orig_mod.``.  Checkpoints must store plain keys (portable across
    compiled / non-compiled loads and consumable by ``llm-serve``); this
    helper also makes legacy checkpoints that accidentally stored the
    prefixed form loadable.
    """
    if any(key.startswith("_orig_mod.") for key in state_dict):
        return {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    return state_dict


def _align_state_dict_keys(state_dict: dict[str, Any], expected_keys: Iterable[str]) -> dict[str, Any]:
    """Remap a checkpoint's keys onto the namespace a wrapped module expects.

    An FSDP ``FULL_STATE_DICT`` ``state_dict()``/``load_state_dict`` names its
    keys after the *wrapped* module: plain when wrapping a bare model,
    ``_orig_mod.*`` when the engine compiled the model before wrapping (FSDP
    over ``torch.compile``).  Checkpoints are stored plain, so on FSDP load
    we must add the ``_orig_mod.`` prefix back when (and only when) the live
    model exposes it (round-76 deep-dive D1).  A no-op when the namespaces
    already agree, and it also accepts the legacy prefixed form.
    """
    prefixed_expected = any(key.startswith("_orig_mod.") for key in expected_keys)
    prefixed_state = any(key.startswith("_orig_mod.") for key in state_dict)
    if prefixed_state and not prefixed_expected:
        return {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    if not prefixed_state and prefixed_expected:
        return {f"_orig_mod.{key}": value for key, value in state_dict.items()}
    return state_dict


def model_for_checkpoint_io(model: nn.Module) -> nn.Module:
    """Return the module that should receive load_state_dict during resume."""
    if isinstance(model, DistributedDataParallel):
        model = model.module
    if hasattr(model, "_orig_mod"):  # torch.compile wrapper
        model = cast(nn.Module, model._orig_mod)
    return model


def _fsdp_state_dict_setup(
    state_dict_type: StateDictType,
) -> tuple[Any, Any]:
    """Return the FSDP ``StateDictType`` and matching config object.

    The FSDP context manager (``FullyShardedDataParallel.state_dict_type``)
    requires both an enum value and a config object that matches it. We
    centralise the mapping here so the two helpers stay in sync.
    """
    from torch.distributed.fsdp import StateDictType
    from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig

    if state_dict_type == "full":
        return StateDictType.FULL_STATE_DICT, FullStateDictConfig()
    if state_dict_type == "sharded":
        # Default sharded config: every rank loads / saves its own shard.
        # The checkpoint path is the checkpoint manager's responsibility;
        # this config only controls FSDP's per-rank behaviour.
        return StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig()
    raise ValueError(f"Unknown state_dict_type {state_dict_type!r}; expected 'full' or 'sharded'.")


def load_model_state_dict(
    model: nn.Module,
    state_dict: dict[str, Any],
    state_dict_type: StateDictType = "full",
) -> None:
    """Load weights into bare, DDP, or FSDP-wrapped models.

    Args:
        model: The wrapped model.
        state_dict: The state dict to load. For FSDP ``"sharded"``
            this should already be a per-rank shard — the framework
            doesn't redistribute it.
        state_dict_type: FSDP only — ``"full"`` (default) for
            single-rank save/load, ``"sharded"`` for memory-bounded
            multi-rank save/load. Ignored for DDP / bare models.
    """
    if is_pp(model):
        # A pipeline stage owns a DISJOINT slice of the full model; the
        # checkpoint carries global model names, so each rank scatters its
        # own slice from the full dict (RIL DEC-049 / TASK-210).
        cast(Any, model)._pp.scatter_load(_strip_compile_prefix(state_dict))
        return
    if model.__class__.__name__ == "FullyShardedDataParallel":
        from torch.distributed.fsdp import FullyShardedDataParallel

        sdt, cfg = _fsdp_state_dict_setup(state_dict_type)
        with FullyShardedDataParallel.state_dict_type(model, sdt, cfg):
            expected = list(model.state_dict().keys())
            model.load_state_dict(_align_state_dict_keys(_strip_compile_prefix(state_dict), expected))
        return
    if is_tp(model):
        # Slice the full state dict into this rank's shards, in place
        # (RIL TASK-200 / DEC-045 — TP checkpoint boundary mirrors FSDP-full).
        cast(Any, model)._tp.scatter_load_state_dict(model_for_checkpoint_io(model), _strip_compile_prefix(state_dict))
        return
    model_for_checkpoint_io(model).load_state_dict(_strip_compile_prefix(state_dict))


def is_fsdp(model: nn.Module) -> bool:
    """True when ``model`` is an FSDP wrapper (by class name, so no torch-dep).

    Used to decide whether a checkpoint save is a cross-rank collective:
    FSDP's ``FULL_STATE_DICT`` ``state_dict()`` all-gathers every shard, so
    every rank must enter it or rank 0 blocks forever (RIL ISS-186).
    """
    return model.__class__.__name__ == "FullyShardedDataParallel"


def is_tp(model: nn.Module) -> bool:
    """True when ``model`` has been tensor-parallelised.

    ``apply_tensor_parallel`` tags the (possibly ``torch.compile``-wrapped)
    model with ``_tp`` metadata; the tensor distribution collectives make a
    checkpoint save a cross-rank operation exactly like FSDP's full state
    dict.
    """
    return getattr(model, "_tp", None) is not None


def allreduce_dp_grads(model: nn.Module) -> None:
    """Make every parameter gradient coherent across a TP model's groups.

    Two reductions, both required for TP models to stay in lockstep:

    * **DP group (2D, RIL TASK-202):** each data-parallel group trains on a
      different data shard, so every rank's gradient (for its shard copies AND
      the replicated parameters) only carries THAT group's shard gradient.
      Averaging over the DP group converges to the true full-batch gradient
      (DDP semantics) — otherwise the identical shard copies held by different
      DP groups diverge.
    * **TP group (replicated params only):** the replicated parameters
      (embedding, norms, row-parallel biases — ``_tp.partition`` axis ``None``)
      must use a BIT-IDENTICAL gradient on every TP rank. They are not
      sharded, so their per-rank values differ by small floating-point drift
      that compounds across steps and silently desynchronises the TP-group
      replicas (surfaced by the bit-exact state-dict check in the 2D engine
      e2e; also latent in pure-TP v1 where parity was only asserted CLOSE).
      Averaging them over the TP group forces the replicas to step together.

    Sharded (column/row-parallel) weights are NOT reduced over the TP group —
    each rank owns a disjoint slice whose gradient belongs only to it.

    Runs AFTER the last gradient contribution of a step (post ``unscale_`` so
    the reduce happens in fp32, post partial-window re-scaling) and BEFORE grad
    clipping / ``optimizer.step``. It MUST NOT run on gradient-accumulation
    interior microbatches (that is exactly what DDP's ``no_sync()`` avoids);
    the engine gates it on step boundaries. A no-op for models that are not
    tensor-parallel.
    """
    if not is_tp(model):
        return
    tp = cast(Any, model)._tp
    dp_group = getattr(tp, "dp_group", None)
    for name, param in model.named_parameters():
        if param.grad is None and not (tp.is_expert_param(name) or tp.is_gate_param(name)):
            continue
        if param.grad is None:
            # MoE router/expert with zero routed tokens (TASK-207): the
            # gradient is genuinely ZERO, but the collective below must be
            # entered by EVERY rank — a peer whose experts got hits enters the
            # gate-SUM / DP-group reduce, so a skipped call would deadlock.
            # Materialising zeros keeps the reduction uniform and is the
            # correct contribution (find_unused_parameters semantics).
            param.grad = torch.zeros_like(param)
        if dp_group is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=dp_group)
        if tp.is_expert_param(name):
            # Expert-parallel shard (TASK-207): each rank owns a DIFFERENT
            # expert, so its gradient belongs only to it — never averaged over
            # the TP group (that would bleed one expert's gradient into its
            # neighbours'). DP-group averaging above still applies: the same
            # expert shards held by different data groups train different data.
            continue
        if tp.is_gate_param(name):
            # Replicated MoE router (TASK-207): unlike norms/embedding, the
            # gate's weight gradient is a per-rank PARTIAL (each rank backprops
            # only ITS experts' contributions through the routing marginals),
            # so the true gradient is the SUM over the TP group — the input
            # all-reduces can never complete it.
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=tp.group)
            continue
        if tp.partition.get(name) is None:
            # Replicated parameter (or full bias of a row-parallel linear):
            # force a bit-identical gradient across the TP group.
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=tp.group)


def clip_grad_norm_tp(
    model: nn.Module,
    max_norm: float,
    group: dist.ProcessGroup | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Clip gradients by the GLOBAL full-model L2 norm, TP-aware.

    ``torch.nn.utils.clip_grad_norm_`` computes the total norm over THIS
    rank's parameters only — under tensor parallelism each rank holds a
    DISJOINT shard, so every rank would clip with a different (too small)
    local factor and the shards drift apart (RIL ISS-253). This helper
    instead all-reduces the per-rank squared norms over the TP ``group`` so
    every rank uses the SAME full-model norm, then clips its locals with the
    common factor (Megatron semantics).

    The per-rank squared sums are already identical across DP groups (the DP
    gradient average ran first), so one TP-group reduce yields the correct
    global norm on every rank. A no-op on the result when ``group`` is ``None``
    (plain ``clip_grad_norm_`` semantics, single-process callers).

    Args:
        model: The (TP-partitioned) model.
        max_norm: Maximum global L2 norm. Gradient values are not modified
            when the global norm stays below it.
        group: The group over which the norm is global (the TP group).
            Defaults to ``model._tp.group``.
        eps: Denominator guard for the clip coefficient.

    Returns:
        The GLOBAL total norm as a 0-dim float32 tensor (used verbatim like
        ``clip_grad_norm_``'s return: ``torch.isfinite`` detects a skipped
        step, ``.item()`` feeds perf monitoring). ``inf``/``NaN`` when any
        rank's gradient is non-finite.
    """
    if group is None and is_tp(model):
        group = cast(Any, model)._tp.group
    params = [p for p in model.parameters() if p.grad is not None]
    # Device comes from the model's parameters, NOT this rank's grads: on a
    # step where every grad is absent (a detached/constant loss), params still
    # tell us where the group lives — the all-reduce below then contributes a
    # proper ZERO on the right device instead of a CPU tensor crashing NCCL
    # with "No backend type associated with device type cpu" (surfaced by the
    # vacuous SFT-loss test during the ISS-253 milestone).
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    total_sq = torch.zeros((), dtype=torch.float32, device=device)
    # ``params`` was filtered to only those with a gradient above, but ty
    # cannot narrow ``Parameter.grad``'s ``Tensor | None`` through the list
    # comprehension — re-guard here so the attribute access stays typed.
    for p in params:
        grad = p.grad
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            # A non-finite contribution poisons the global norm (and the step
            # must be skipped); propagate inf/NaN so the caller's isfinite
            # check behaves exactly like clip_grad_norm_'s.
            total_sq = torch.tensor(float("inf"), dtype=torch.float32, device=device)
            break
        total_sq += (grad.float() * grad.float()).sum()
    if group is not None:
        dist.all_reduce(total_sq, op=dist.ReduceOp.SUM, group=group)
    total_norm = total_sq.sqrt()
    clip_coef = max_norm / (total_norm + eps)
    if bool(clip_coef < 1.0):
        for p in params:
            grad = p.grad
            if grad is not None:
                grad.mul_(clip_coef)
    return total_norm


def model_state_dict(
    model: nn.Module,
    state_dict_type: StateDictType = "full",
) -> dict[str, Any]:
    """Extract a plain state dict from wrapped or bare models.

    Args:
        model: The wrapped model.
        state_dict_type: FSDP only — see :func:`load_model_state_dict`.
    """
    if isinstance(model, DistributedDataParallel):
        model = model.module
    fsdp_state: dict[str, Any] | None = None
    if model.__class__.__name__ == "FullyShardedDataParallel":
        from torch.distributed.fsdp import FullyShardedDataParallel

        sdt, cfg = _fsdp_state_dict_setup(state_dict_type)
        with FullyShardedDataParallel.state_dict_type(model, sdt, cfg):
            fsdp_state = model.state_dict()
    # ``torch.compile`` wraps the module as ``_orig_mod`` and prefixes its
    # state-dict keys; unwrap so checkpoints are portable (llm-serve loads
    # plain module keys and has no torch.compile graph).  Applies to the FSDP
    # FULL_STATE_DICT too: the engine compiles the model *before* wrapping it
    # in FSDP, so an FSDP full-state dict carries ``_orig_mod.*`` keys just
    # like a bare compiled module, and a checkpoint missing the strip cannot
    # be resumed (load always normalizes, and FSDP expects ``_orig_mod.*``)
    # nor transferred to ``llm-serve`` (round-76 deep-dive D1).
    if is_pp(model):
        # Cross-stage all-gather: EVERY rank must enter it (mirrors the TP /
        # FSDP rule — RIL ISS-186) so rank 0's save does not block. Each rank
        # contributes its owned slice under GLOBAL model names; the manager
        # writes rank 0's merged dict, and a resume scatters it back.
        return cast(Any, model)._pp.full_state_dict()
    if is_tp(model):
        # Cross-rank all-gather: EVERY rank must enter it (mirrors the FSDP
        # FULL_STATE_DICT rule — RIL ISS-186) so rank 0's save does not block.
        # Each rank rebuilds the full dict; the manager writes rank 0's.
        return cast(Any, model)._tp.gather_full_state_dict(model_for_checkpoint_io(model))
    state = fsdp_state if fsdp_state is not None else model_for_checkpoint_io(model).state_dict()
    return _strip_compile_prefix(state)
