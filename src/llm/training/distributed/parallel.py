"""Parallel training strategy helpers (DDP, FSDP)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def wrap_model_for_training(
    model: nn.Module,
    *,
    parallel_strategy: str,
    device: torch.device,
    world_size: int,
) -> nn.Module:
    """Wrap a model for distributed training when running on CUDA."""
    if world_size <= 1 or device.type != "cuda":
        return model

    if parallel_strategy == "ddp":
        return DDP(model, device_ids=[device.index], find_unused_parameters=False)

    if parallel_strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return FSDP(model, device_id=device.index)

    raise ValueError(f"Unknown parallel_strategy '{parallel_strategy}'. Expected 'ddp' or 'fsdp'.")


def model_for_checkpoint_io(model: nn.Module) -> nn.Module:
    """Return the module that should receive load_state_dict during resume."""
    if isinstance(model, DDP):
        return model.module
    return model


def load_model_state_dict(model: nn.Module, state_dict: dict[str, Any]) -> None:
    """Load weights into bare, DDP, or FSDP wrapped models."""
    if model.__class__.__name__ == "FullyShardedDataParallel":
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig()):
            model.load_state_dict(state_dict)
        return
    model_for_checkpoint_io(model).load_state_dict(state_dict)


def model_state_dict(model: nn.Module) -> dict[str, Any]:
    """Extract a plain state dict from wrapped or bare models."""
    if isinstance(model, DDP):
        return model.module.state_dict()
    if model.__class__.__name__ == "FullyShardedDataParallel":
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig()):
            return model.state_dict()
    return model.state_dict()
