import argparse
import logging
import sys

import torch
import torch.multiprocessing as mp

from llm.training.core.config import Config
from llm.training.core.engine import TrainingEngine
from llm.training.core.utils import DistributedManager, Logger
from llm.training.tasks.classification_task import ClassificationTask

# --- Import your tasks here ---
from llm.training.tasks.regression_task import RegressionTask

# --- Map task names to task classes ---
AVAILABLE_TASKS = {
    "regression": RegressionTask,
    "classification": ClassificationTask,
}


def train_worker(rank: int, world_size: int, config: Config, task_class):
    """The worker function for each DDP process."""
    distributed_manager = DistributedManager(config.distributed)
    try:
        distributed_manager.setup(rank, world_size)
        task = task_class(config)
        engine = TrainingEngine(config, task, rank, world_size)
        engine.run()
    except Exception:
        logging.getLogger().exception(f"An error occurred in rank {rank}")
        raise
    finally:
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
    sys.argv = [sys.argv[0]] + remaining_argv

    config = Config.from_args_and_env()
    task_class = AVAILABLE_TASKS[args.task]

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    distributed_manager = DistributedManager(config.distributed)
    world_size = distributed_manager.get_world_size()

    logger = Logger(0, config.logging)  # Main process logger

    if torch.cuda.is_available() and world_size > 0:
        logger.info(f"Selected Task: {args.task}")
        # ... (logging config info can go here)

        if world_size > 1:
            logger.info(f"🚀 Spawning {world_size} DDP processes...")
            mp.spawn(
                train_worker, args=(world_size, config, task_class), nprocs=config.distributed.gpus_per_node, join=True
            )
        else:
            logger.info("🚀 Starting single-GPU training...")
            train_worker(0, 1, config, task_class)
    else:
        logger.error("❌ No GPUs found or world_size is zero. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
