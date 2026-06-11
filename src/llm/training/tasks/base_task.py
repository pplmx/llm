import abc
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

from llm.data.base import BaseDataModule
from llm.runtime.checkpoint import CheckpointContributor
from llm.training.core.config import Config

if TYPE_CHECKING:
    from llm.training.core.engine import TrainingEngine


class TrainingTask(abc.ABC, CheckpointContributor):
    """
    An abstract base class for defining a training task.

    Standard tasks implement build_* + train_step and run through TrainingEngine.
    Custom-loop tasks (e.g. PPO) set ``uses_standard_loop = False`` and override
    ``run_training``.
    """

    uses_standard_loop: bool = True

    def __init__(self, config: Config, data_module: BaseDataModule):
        self.config = config
        self.data_module = data_module

    def uses_standard_training_loop(self) -> bool:
        return self.uses_standard_loop

    def prepare_training(self, engine: TrainingEngine) -> None:
        """Hook for custom-loop tasks after the model is on device."""

    def get_resume_optimizer(self) -> optim.Optimizer | None:
        """Return the optimizer used when resuming a custom training loop."""
        return None

    def get_checkpoint_state(self) -> dict[str, Any] | None:
        return None

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        pass

    def run_training(self, engine: TrainingEngine) -> None:
        """Execute a non-standard training loop."""
        raise NotImplementedError(f"{type(self).__name__} must implement run_training().")

    @abc.abstractmethod
    def build_model(self) -> nn.Module:
        """Builds and returns the model for the task."""
        pass

    @abc.abstractmethod
    def build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Builds and returns the optimizer."""
        pass

    @abc.abstractmethod
    def build_scheduler(self, optimizer: optim.Optimizer) -> LRScheduler | None:
        """Builds and returns the learning rate scheduler."""
        pass

    @abc.abstractmethod
    def build_criterion(self) -> nn.Module:
        """Builds and returns the loss function."""
        pass

    @abc.abstractmethod
    def train_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        """
        Performs a single training step.

        Returns:
            A tuple of (loss tensor, metrics dict).
        """
        pass

    @abc.abstractmethod
    def validation_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        """Performs a single validation step."""
        pass
