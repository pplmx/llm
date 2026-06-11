import logging
import sys
from pathlib import Path

import click
import torch
import torch.multiprocessing as mp
import typer
from rich.logging import RichHandler

from llm.training.core.callbacks import LRSchedulerCallback, MetricsLogger, TensorBoardLogger
from llm.training.core.config import Config
from llm.training.core.engine import TrainingEngine
from llm.training.core.utils import DistributedManager
from llm.training.task_registry import TASK_REGISTRY
from llm.training.tasks import builtin as _task_registry  # noqa: F401 — register built-in tasks

# --- Typer App ---
app = typer.Typer(pretty_exceptions_show_locals=False)


def configure_logging(log_level: str = "INFO"):
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


def train_worker(rank: int, world_size: int, config: Config, task_name: str):
    """The worker function for each DDP process."""
    distributed_manager = DistributedManager(config.distributed)
    try:
        distributed_manager.setup(rank, world_size)

        task_spec = TASK_REGISTRY.get(task_name)
        data_module = task_spec.data_module_factory(config)
        data_module.prepare_data()
        data_module.setup()

        task = task_spec.task_cls(config, data_module)

        callbacks = [
            MetricsLogger(),
            TensorBoardLogger(log_dir=config.logging.log_dir),
            LRSchedulerCallback(),
        ]

        engine = TrainingEngine(
            config,
            task,
            rank,
            world_size,
            data_module=data_module,
            callbacks=callbacks,
        )
        engine.run()
    except Exception:
        logging.getLogger().exception(f"An error occurred in rank {rank}")
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
        click_type=click.Choice(TASK_REGISTRY.names(), case_sensitive=True),
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

    logger.info(f"Selected Task: {task}")
    logger.info(f"Determined world_size: {world_size}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}, Count: {torch.cuda.device_count()}")

    if world_size > 1:
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= world_size):
            logger.error(f"❌ DDP Error: world_size={world_size}, but available GPUs={torch.cuda.device_count()}.")
            sys.exit(1)

        logger.info(f"🚀 Spawning {world_size} DDP processes...")
        mp.spawn(train_worker, args=(world_size, config, task), nprocs=world_size, join=True)
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
