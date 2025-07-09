# Training Framework Usage Guide

This guide provides an overview of the training framework and how to use it for your machine learning projects.

## 1. Introduction to the Training Framework

The training framework is designed to provide a modular and extensible way to train deep learning models. It separates concerns into distinct components:

-   **Config**: Manages all training parameters and hyperparameters.
-   **DataModule**: Handles data loading, preprocessing, and provides DataLoaders.
-   **TrainingTask**: Defines the model, optimizer, loss function, and training/validation steps for a specific task.
-   **TrainingEngine**: Orchestrates the training loop, handles distributed training, and integrates with callbacks.
-   **Callbacks**: Allow injecting custom logic at various points during the training process.

## 2. How to Define a New Task (`TrainingTask` subclass)

To define a new training task, you need to create a class that inherits from `llm.training.tasks.base_task.TrainingTask` and implement its abstract methods:

-   `build_model(self) -> nn.Module`: Define your model architecture.
-   `build_optimizer(self, model: nn.Module) -> optim.Optimizer`: Define the optimizer for your model.
-   `build_scheduler(self, optimizer: optim.Optimizer) -> LRScheduler | None`: Define the learning rate scheduler (optional).
-   `build_criterion(self) -> nn.Module`: Define the loss function.
-   `train_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]`: Implement a single training step.
-   `validation_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]`: Implement a single validation step.

**Example (`llm/training/tasks/regression_task.py`):**

```python
# (Content of regression_task.py will go here)
```

## 3. How to Define a New Data Module (`BaseDataModule` subclass)

To handle your data, create a class that inherits from `llm.data.data_module.BaseDataModule` and implement its abstract methods:

-   `prepare_data(self)`: For downloading or preparing raw data (called once per node).
-   `setup(self, stage: Optional[str] = None)`: For loading and splitting data (called on every GPU).
-   `train_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler]`: Returns the training DataLoader and its sampler.
-   `val_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler]`: Returns the validation DataLoader and its sampler.

**Example (`llm/data/synthetic_data_module.py`):**

```python
# (Content of synthetic_data_module.py will go here)
```

## 4. How to Configure Training (`Config` class and YAML/CLI)

The `llm.training.core.config.Config` class manages all training parameters. You can define your configuration using:

-   **Default values**: Defined in the `Config` and its nested dataclasses.
-   **Command-line arguments**: Overrides default values (e.g., `--epochs 20`).
-   **YAML files**: Load and save configurations for reproducibility.

**Example (Loading from YAML and saving):**

```python
from llm.training.core.config import Config

# Load config from a YAML file
my_config = Config.from_yaml("path/to/my_config.yaml")

# Override some parameters via CLI (handled automatically by from_args_and_env)
# e.g., python train.py --epochs 20 --training-batch-size 64

# Save current config to a new YAML file
my_config.save_to_yaml("path/to/saved_config.yaml")
```

## 5. How to Use Callbacks

Callbacks allow you to inject custom logic into the training loop. The framework provides several built-in callbacks:

-   `MetricsLogger`: Logs training and validation metrics at the end of each epoch.
-   `EarlyStopping`: Stops training early if a monitored metric does not improve.
-   `TensorBoardLogger`: Logs metrics and other data to TensorBoard for visualization.
-   `LRSchedulerCallback`: Logs learning rate at each step and epoch end.

To use callbacks, instantiate them and pass a list of callback instances to the `TrainingEngine`.

**Example (`llm/training/train.py` snippet):**

```python
from llm.training.core.callbacks import MetricsLogger, EarlyStopping, TensorBoardLogger, LRSchedulerCallback

# ... inside train_worker function ...

        callbacks = [
            MetricsLogger(),
            EarlyStopping(monitor="val_loss", patience=5),
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
```

## 6. Running Training (`train.py`)

The main entry point for training is `llm/training/train.py`. You can run it from the command line:

```bash
python -m llm.training.train --task regression --epochs 10 --training-batch-size 32
```

For distributed training, use `torchrun` or `torch.distributed.launch`:

```bash
torchrun --nproc_per_node=2 llm/training/train.py --task regression
```

Remember to adjust the `--task` argument and other configuration parameters as needed.
