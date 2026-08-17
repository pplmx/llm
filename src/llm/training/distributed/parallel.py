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
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

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
) -> nn.Module:
    """Wrap a model for distributed training.

    For single-rank or CPU runs the model is returned unchanged —
    wrapping a CPU model in ``DistributedDataParallel`` is a no-op
    that can confuse some optimisers, and ``FullyShardedDataParallel``
    requires CUDA.

    Args:
        model: The bare ``nn.Module`` to wrap.
        parallel_strategy: ``"ddp"`` or ``"fsdp"``.
        device: Target device (used to set ``device_ids`` for DDP
            and ``device_id`` for FSDP).
        world_size: Number of ranks in the process group.
        fsdp_mixed_precision: FSDP parameter / gradient dtype.
            One of ``"fp32"``, ``"bf16"``, ``"fp16"``.
        fsdp_auto_wrap_min_params: FSDP size-based auto-wrap
            threshold. ``0`` disables auto-wrap.
        fsdp_cpu_offload: Offload FSDP params to CPU when idle.

    Raises:
        ValueError: if ``parallel_strategy`` is not recognised.
        RuntimeError: if FSDP is requested on a CPU-only host
            (FSDP needs CUDA + a process group).
    """
    if world_size <= 1 or device.type != "cuda":
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

    raise ValueError(f"Unknown parallel_strategy '{parallel_strategy}'. Expected 'ddp' or 'fsdp'.")


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
    if model.__class__.__name__ == "FullyShardedDataParallel":
        from torch.distributed.fsdp import FullyShardedDataParallel

        sdt, cfg = _fsdp_state_dict_setup(state_dict_type)
        with FullyShardedDataParallel.state_dict_type(model, sdt, cfg):
            expected = list(model.state_dict().keys())
            model.load_state_dict(_align_state_dict_keys(_strip_compile_prefix(state_dict), expected))
        return
    model_for_checkpoint_io(model).load_state_dict(_strip_compile_prefix(state_dict))


def is_fsdp(model: nn.Module) -> bool:
    """True when ``model`` is an FSDP wrapper (by class name, so no torch-dep).

    Used to decide whether a checkpoint save is a cross-rank collective:
    FSDP's ``FULL_STATE_DICT`` ``state_dict()`` all-gathers every shard, so
    every rank must enter it or rank 0 blocks forever (RIL ISS-186).
    """
    return model.__class__.__name__ == "FullyShardedDataParallel"


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
    state = fsdp_state if fsdp_state is not None else model_for_checkpoint_io(model).state_dict()
    return _strip_compile_prefix(state)
