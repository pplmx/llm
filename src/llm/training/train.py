import argparse
import logging
import sys

import torch
import torch.multiprocessing as mp

from llm.data.synthetic_data_module import SyntheticDataModule  # Added SyntheticDataModule
from llm.training.core.callbacks import LRSchedulerCallback, MetricsLogger, TensorBoardLogger
from llm.training.core.config import Config
from llm.training.core.engine import TrainingEngine
from llm.training.core.utils import DistributedManager, Logger

# --- Import your tasks here ---
from llm.training.tasks.regression_task import RegressionTask

# --- Map task names to task classes ---
AVAILABLE_TASKS = {
    "regression": RegressionTask,
}


def train_worker(rank: int, world_size: int, config: Config, task_class):
    """The worker function for each DDP process."""
    distributed_manager = DistributedManager(config.distributed)
    try:
        distributed_manager.setup(rank, world_size)

        # Instantiate DataModule first
        data_module = SyntheticDataModule(config)
        data_module.prepare_data()
        data_module.setup()

        # Now instantiate the task with the data_module
        task = task_class(config, data_module)

        # Instantiate callbacks
        callbacks = [
            MetricsLogger(),
            TensorBoardLogger(log_dir=config.logging.log_dir),
            LRSchedulerCallback(),  # Added LRSchedulerCallback
        ]

        engine = TrainingEngine(
            config,
            task,
            rank,
            world_size,
            data_module=data_module,  # Pass data_module
            callbacks=callbacks,  # Pass callbacks
        )
        engine.run()
    except Exception:
        logging.getLogger().exception(f"An error occurred in rank {rank}")
        raise
    finally:
        # Ensure all processes sync up before cleaning up
        if world_size > 1:  # Barrier is only relevant for DDP
            DistributedManager.barrier()
        distributed_manager.cleanup()


def main():
    # We add a new argument to select the task
    parser = argparse.ArgumentParser(description="Modular PyTorch DDP Training Framework")
    parser.add_argument(
        "--task", type=str, required=True, choices=AVAILABLE_TASKS.keys(), help="Name of the task to run."
    )

    # Let the Config class handle the rest of the arguments
    # We parse known args first to get the task, then let Config parse the rest
    args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_argv]

    config = Config.from_args_and_env()
    task_class = AVAILABLE_TASKS[args.task]

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    distributed_manager = DistributedManager(config.distributed)
    world_size = distributed_manager.get_world_size()

    distributed_manager = DistributedManager(config.distributed)
    # get_world_size() now returns 1 if no GPUs, or calculated size if GPUs exist (for single node).
    world_size = distributed_manager.get_world_size()

    # Logger setup: rank 0 for the main process. TrainingEngine will determine actual device.
    logger = Logger(0, config.logging)
    logger.info(f"Selected Task: {args.task}")
    logger.info(f"Determined world_size from DistributedManager: {world_size}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}, GPU Count: {torch.cuda.device_count()}")
    logger.info(
        f"Configured gpus_per_node: {config.distributed.gpus_per_node}, num_nodes: {config.distributed.num_nodes}"
    )

    if world_size > 1:
        # This implies DDP on GPUs. get_world_size() should only return > 1 if GPUs are configured and available.
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= world_size):
            logger.error(
                f"❌ DDP Misconfiguration: world_size is {world_size}, but CUDA not available or "
                f"insufficient GPUs ({torch.cuda.device_count()} available). Exiting."
            )
            sys.exit(1)

        logger.info(f"🚀 Spawning {world_size} DDP processes for GPU training...")
        # For single-node DDP, nprocs should be the world_size determined (number of GPUs to use).
        # The arguments to train_worker will be (rank, world_size, config, task_class).
        mp.spawn(train_worker, args=(world_size, config, task_class), nprocs=world_size, join=True)
    elif world_size == 1:
        # This handles:
        # 1. CPU only (get_world_size returned 1 because no GPUs were found/configured for use)
        # 2. Single GPU (get_world_size returned 1, e.g., gpus_per_node=1 or only 1 GPU available and used)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # Check if this single process run is intended for a GPU
            # TrainingEngine will place on cuda:0 if world_size=1 and rank=0 and cuda is available
            logger.info("🚀 Starting single-process training (GPU available, will use if rank 0 maps to GPU)...")
        else:
            logger.info("🚀 Starting single-process training (CPU)...")
        train_worker(0, 1, config, task_class)  # rank 0, world_size 1
    else:
        # world_size <= 0, which should ideally be prevented by get_world_size returning min 1.
        logger.error(f"❌ Invalid world_size ({world_size}) determined by DistributedManager. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
