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

from typing import Any, Literal

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
        from torch.distributed.fsdp import MixedPrecision

        # FP16 needs a loss scaler; we don't wire one here because
        # AMP is the trainer's responsibility. Recommend BF16 unless
        # the user has a specific reason for FP16.
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
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
    from torch.distributed.fsdp import FullyShardedDataParallel
    from torch.distributed.fsdp.backends.cpu_offload import CPUOffload

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
        return DistributedDataParallel(
            model,
            device_ids=[device.index],
            find_unused_parameters=False,
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


def model_for_checkpoint_io(model: nn.Module) -> nn.Module:
    """Return the module that should receive load_state_dict during resume."""
    if isinstance(model, DistributedDataParallel):
        return model.module
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
            model.load_state_dict(state_dict)
        return
    model_for_checkpoint_io(model).load_state_dict(state_dict)


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
        return model.module.state_dict()
    if model.__class__.__name__ == "FullyShardedDataParallel":
        from torch.distributed.fsdp import FullyShardedDataParallel

        sdt, cfg = _fsdp_state_dict_setup(state_dict_type)
        with FullyShardedDataParallel.state_dict_type(model, sdt, cfg):
            return model.state_dict()
    return model.state_dict()
