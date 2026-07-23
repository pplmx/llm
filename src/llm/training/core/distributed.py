"""Distributed-training setup/teardown helpers."""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

from llm.training.core.config import DistributedConfig

logger = logging.getLogger(__name__)


class DistributedManager:
    """Thin wrapper around ``torch.distributed`` for DDP setup/teardown."""

    def __init__(self, config: DistributedConfig):
        self.config = config

    def setup(self, rank: int, world_size: int):
        torch.manual_seed(42 + rank)  # Seed for CPU operations

        if world_size > 1 and torch.cuda.is_available() and torch.cuda.device_count() > 0:
            os.environ["MASTER_ADDR"] = self.config.master_addr
            os.environ["MASTER_PORT"] = self.config.master_port
            # Ensure backend is NCCL if GPUs are used for DDP
            backend = self.config.backend if self.config.backend else "nccl"
            if backend != "nccl" and torch.cuda.is_available() and backend != "gloo":
                # In a typical DDP setup with GPUs, nccl is preferred.
                # If another backend is specified, respect it but log a warning if it's not 'gloo' for CPU.
                logger.warning(
                    "Non-standard backend '%s' specified for DDP with GPUs. 'nccl' is typical.",
                    backend,
                )

            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

            # Only set device and CUDA seed if CUDA is available and being used
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                torch.cuda.set_device(rank % torch.cuda.device_count())
                torch.cuda.manual_seed_all(42 + rank)
        elif world_size == 1:
            # Single process execution (could be CPU or single GPU without DDP)
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                torch.cuda.manual_seed_all(42 + rank)  # Still seed CUDA if available
            pass  # No DDP setup needed
        else:
            # This case should ideally be prevented by get_world_size logic
            # or handled in the main training script.
            # If world_size is 0 or <0, or >1 but no GPUs, it's an invalid state.
            if not (torch.cuda.is_available() and torch.cuda.device_count() > 0) and world_size > 1:
                logger.warning(
                    "world_size is %d but no GPUs found. DDP setup will be skipped. Check configuration.",
                    world_size,
                )
            # If it still proceeds, it will likely fail later if DDP operations are attempted.
            # Forcing world_size to 1 if no GPUs might be an option in get_world_size.
            pass

    @staticmethod
    def cleanup():
        if dist.is_initialized():
            dist.destroy_process_group()

    def get_world_size(self) -> int:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            # If no GPUs, world size is 1 (CPU execution)
            return 1

        # If GPUs are available, calculate based on config
        # Default to 1 GPU per node if not specified, assuming at least one is available.
        gpus_per_node = (
            self.config.gpus_per_node if self.config.gpus_per_node is not None and self.config.gpus_per_node > 0 else 1
        )
        num_nodes = self.config.num_nodes if self.config.num_nodes > 0 else 1

        calculated_world_size = num_nodes * gpus_per_node

        # Ensure world size does not exceed available GPUs if running on a single node.
        # For multi-node, this check is more complex and typically handled by the cluster manager.
        if num_nodes == 1:
            actual_gpus = torch.cuda.device_count()
            if actual_gpus == 0:  # Should be caught by the first check
                return 1
            if calculated_world_size > actual_gpus:
                # Log a warning or adjust? For now, let's assume config is trusted or user intends to use subset.
                # print(f"Warning: Configured world size ({calculated_world_size}) > available GPUs ({actual_gpus}). Using {actual_gpus}.")
                # return actual_gpus
                pass  # Trusting config for now, but this could be a point of error.
                # The script currently exits if world_size > 0 but no GPUs are found in main.

        return calculated_world_size if calculated_world_size > 0 else 1

    @staticmethod
    def barrier():
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= dist.get_world_size()
        return tensor
