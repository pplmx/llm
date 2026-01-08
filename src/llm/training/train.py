import logging
import sys
from enum import Enum
from pathlib import Path

import torch
import torch.multiprocessing as mp
import typer
from rich.logging import RichHandler

from llm.data.dpo_data_module import DPODataModule
from llm.data.sft_data_module import SFTDataModule
from llm.data.synthetic_data_module import SyntheticDataModule
from llm.data.text_data_module import TextDataModule
from llm.training.core.callbacks import LRSchedulerCallback, MetricsLogger, TensorBoardLogger
from llm.training.core.config import Config
from llm.training.core.engine import TrainingEngine
from llm.training.core.utils import DistributedManager
from llm.training.tasks.dpo_task import DPOTask
from llm.training.tasks.lm_task import LanguageModelingTask
from llm.training.tasks.regression_task import RegressionTask
from llm.training.tasks.sft_task import SFTTask

# --- Typer App ---
app = typer.Typer(pretty_exceptions_show_locals=False)


# --- Task Enum for Typer choices ---
class TaskName(str, Enum):
    regression = "regression"
    lm = "lm"
    sft = "sft"
    dpo = "dpo"


# --- Map task names to task classes ---
AVAILABLE_TASKS = {
    TaskName.regression: RegressionTask,
    TaskName.lm: LanguageModelingTask,
    TaskName.sft: SFTTask,
    TaskName.dpo: DPOTask,
}


def configure_logging(log_level: str = "INFO"):
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


def train_worker(rank: int, world_size: int, config: Config, task_class):
    """The worker function for each DDP process."""
    distributed_manager = DistributedManager(config.distributed)
    try:
        distributed_manager.setup(rank, world_size)

        if task_class == LanguageModelingTask:
            data_module = TextDataModule(config)
        elif task_class == SFTTask:
            data_module = SFTDataModule(config)
        elif task_class == DPOTask:
            data_module = DPODataModule(config)
        else:
            data_module = SyntheticDataModule(config)
        data_module.prepare_data()
        data_module.setup()

        # Now instantiate the task with the data_module
        task = task_class(config, data_module)

        # Instantiate callbacks
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
    task: TaskName = typer.Option(..., help="Name of the task to run."),
    config_path: Path | None = typer.Option(None, help="Path to YAML config file."),
    epochs: int | None = typer.Option(None, help="Override training epochs"),
    batch_size: int | None = typer.Option(None, help="Override batch size"),
    lr: float | None = typer.Option(None, help="Override learning rate"),
    num_samples: int | None = typer.Option(None, help="Override number of synthetic samples"),
    compile: bool = typer.Option(True, help="Enable torch.compile"),
    amp: bool = typer.Option(True, help="Enable AMP"),
):
    """
    Modular PyTorch DDP Training Framework.
    """
    configure_logging()
    logger = logging.getLogger("train")

    # 1. Load Base Config
    # 1. Load Base Config
    config = Config.from_yaml(config_path) if config_path else Config()

    # 2. Apply CLI Overrides
    if epochs is not None:
        config.training.epochs = epochs
    if batch_size is not None:
        config.training.batch_size = batch_size
    if lr is not None:
        config.training.lr = lr
    if num_samples is not None:
        config.training.num_samples = num_samples

    # Apply boolean flags
    config.optimization.use_compile = compile
    config.optimization.use_amp = amp

    # Re-validate
    # Note: Pydantic models validate on assignment if ConfigDict(validate_assignment=True) is set,
    # but our nested structure makes it simpler to just let it be or manually re-trigger if needed.

    task_class = AVAILABLE_TASKS[task]

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    distributed_manager = DistributedManager(config.distributed)
    # get_world_size() now returns 1 if no GPUs, or calculated size if GPUs exist.
    world_size = distributed_manager.get_world_size()

    # Logger setup
    logger.info(f"Selected Task: {task.value}")
    logger.info(f"Determined world_size: {world_size}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}, Count: {torch.cuda.device_count()}")

    if world_size > 1:
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= world_size):
            logger.error(f"❌ DDP Error: world_size={world_size}, but available GPUs={torch.cuda.device_count()}.")
            sys.exit(1)

        logger.info(f"🚀 Spawning {world_size} DDP processes...")
        mp.spawn(train_worker, args=(world_size, config, task_class), nprocs=world_size, join=True)
    elif world_size == 1:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            logger.info("🚀 Single-process GPU training...")
        else:
            logger.info("🚀 Single-process CPU training...")
        train_worker(0, 1, config, task_class)
    else:
        logger.error(f"❌ Invalid world_size ({world_size}).")
        sys.exit(1)


if __name__ == "__main__":
    app()
