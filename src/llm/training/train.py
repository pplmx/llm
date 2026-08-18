import logging
import sys
from pathlib import Path
from typing import Any, cast

import click
import torch
import torch.multiprocessing as mp
import typer
from rich.logging import RichHandler

from llm.runtime.plugins import load_entry_point_hooks
from llm.training.core.callbacks import Callback, LRSchedulerCallback, MetricsLogger, TensorBoardLogger
from llm.training.core.config import Config
from llm.training.core.engine import TrainingEngine
from llm.training.core.utils import DistributedManager
from llm.training.task_registry import TASK_REGISTRY
from llm.training.tasks import builtin as _task_registry  # noqa: F401 — register built-in tasks

load_entry_point_hooks("llm.tasks")

# --- Typer App ---
app = typer.Typer(pretty_exceptions_show_locals=False)


def configure_logging(log_level: str = "INFO"):
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


def _global_rank(node_rank: int, local_rank: int, local_world_size: int) -> int:
    """Global process-group rank for a worker on a node (RIL TASK-191).

    Every node spawns ``gpus_per_node`` local workers; the global rank
    ``node_rank * local_world_size + local_rank`` makes them all rendezvous
    into ONE process group without the per-node rank collisions that broke
    multi-node launch (ISS-229).
    """
    return node_rank * local_world_size + local_rank


def train_worker(
    local_rank: int,
    world_size: int,
    config: Config,
    task_name: str,
    node_rank: int = 0,
    local_world_size: int | None = None,
):
    """The worker function for each DDP process.

    ``local_rank`` is the GPU-local index on this node; the process-group
    identity is the GLOBAL rank (see :func:`_global_rank`).
    """
    global_rank = _global_rank(
        node_rank,
        local_rank,
        local_world_size if local_world_size is not None else world_size,
    )
    distributed_manager = DistributedManager(config.distributed)
    try:
        distributed_manager.setup(global_rank, world_size, local_rank=local_rank)

        task_spec = TASK_REGISTRY.get(task_name)
        data_module = task_spec.data_module_factory(config)
        data_module.prepare_data()
        data_module.setup()

        task = task_spec.task_cls(config, data_module)

        callbacks: list[Callback] = [
            MetricsLogger(),
            TensorBoardLogger(log_dir=config.logging.log_dir),
            LRSchedulerCallback(),
        ]

        engine = TrainingEngine(
            config,
            task,
            global_rank,
            world_size,
            data_module=data_module,
            callbacks=callbacks,
            local_rank=local_rank,
        )
        engine.run()
    except Exception:
        logging.getLogger().exception(f"An error occurred in rank {global_rank}")
        raise
    finally:
        if world_size > 1:
            DistributedManager.barrier()
        distributed_manager.cleanup()


@app.command()
def main(
    task: str = typer.Option(
        ...,
        help="Name of the task to run.",
        click_type=cast(Any, click.Choice(TASK_REGISTRY.names(), case_sensitive=True)),
    ),
    config_path: Path | None = typer.Option(None, help="Path to YAML config file."),
    epochs: int | None = typer.Option(None, help="Override training epochs"),
    batch_size: int | None = typer.Option(None, help="Override batch size"),
    lr: float | None = typer.Option(None, help="Override learning rate"),
    num_samples: int | None = typer.Option(None, help="Override number of synthetic samples"),
    steps_per_epoch: int | None = typer.Option(None, help="Override streaming steps per epoch"),
    compile: bool = typer.Option(True, help="Enable torch.compile"),
    amp: bool = typer.Option(True, help="Enable AMP"),
):
    """
    Modular PyTorch DDP Training Framework.
    """
    configure_logging()
    logger = logging.getLogger("train")

    config = Config.from_yaml(config_path) if config_path else Config()

    if epochs is not None:
        config.training.epochs = epochs
    if batch_size is not None:
        config.training.batch_size = batch_size
    if lr is not None:
        config.training.lr = lr
    if num_samples is not None:
        config.training.num_samples = num_samples
    if steps_per_epoch is not None:
        config.data.steps_per_epoch = steps_per_epoch

    config.optimization.use_compile = compile
    config.optimization.use_amp = amp

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    distributed_manager = DistributedManager(config.distributed)
    world_size = distributed_manager.get_world_size()
    node_rank = config.distributed.node_rank
    # Each node spawns only its OWN ``gpus_per_node`` workers; the global
    # process-group rank offsets by ``node_rank * local_world_size`` so
    # multiple nodes join one group without collisions (RIL TASK-191/ISS-229).
    # Single-node keeps node_rank=0 and local == world (identical behaviour).
    gpus_per_node = config.distributed.gpus_per_node
    local_world_size = gpus_per_node if (gpus_per_node is not None and gpus_per_node > 0) else 1

    logger.info(f"Selected Task: {task}")
    logger.info(f"Determined world_size: {world_size} (node {node_rank}, local workers {local_world_size})")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}, Count: {torch.cuda.device_count()}")

    if world_size > 1:
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= local_world_size):
            logger.error(
                f"❌ DDP Error: world_size={world_size}, but each node only has "
                f"GPUs={torch.cuda.device_count()} for local_world_size={local_world_size}."
            )
            sys.exit(1)

        logger.info(f"🚀 Node {node_rank}: spawning {local_world_size} DDP processes (global world {world_size})...")
        mp.spawn(
            train_worker,
            args=(world_size, config, task, node_rank, local_world_size),
            nprocs=local_world_size,
            join=True,
        )
    elif world_size == 1:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            logger.info("🚀 Single-process GPU training...")
        else:
            logger.info("🚀 Single-process CPU training...")
        train_worker(0, 1, config, task)
    else:
        logger.error(f"❌ Invalid world_size ({world_size}).")
        sys.exit(1)


if __name__ == "__main__":
    app()
