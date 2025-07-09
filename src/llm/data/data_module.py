import abc
from typing import Any  # Added Any

from torch.utils.data import DataLoader, DistributedSampler


class BaseDataModule(abc.ABC):
    """
    Abstract base class for defining a DataModule.

    A DataModule encapsulates all the steps needed to process data:
    - Downloading/preparing data (prepare_data)
    - Loading/splitting data (setup)
    - Creating DataLoaders for training and validation.
    """

    def __init__(self, config: Any):
        self.config = config

    @abc.abstractmethod
    def prepare_data(self):
        """Download or prepare data. Only called once per node in DDP."""
        pass

    @abc.abstractmethod
    def setup(self, stage: str | None = None):
        """Load and split data. Called on every GPU. Stage can be 'fit', 'validate', 'test', 'predict'."""
        pass

    @abc.abstractmethod
    def train_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler]:
        """Returns the DataLoader and DistributedSampler for training."""
        pass

    @abc.abstractmethod
    def val_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler]:
        """Returns the DataLoader and DistributedSampler for validation."""
        pass
