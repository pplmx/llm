import abc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, DistributedSampler

from llm.training.core.config import Config


class TrainingTask(abc.ABC):
    """
    An abstract base class for defining a training task.
    Users should subclass this and implement the abstract methods.
    """

    def __init__(self, config: Config):
        self.config = config

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
    def build_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler]:
        """Builds and returns the dataloader and its distributed sampler."""
        pass

    @abc.abstractmethod
    def train_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        """
        Performs a single training step.

        Args:
            batch: The data batch from the dataloader.
            model: The model to train.
            criterion: The loss function.

        Returns:
            A tuple containing:
            - The loss tensor for backpropagation.
            - A dictionary of metrics to log (e.g., {'loss': loss.item()}).
        """
        pass
