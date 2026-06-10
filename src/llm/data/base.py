import abc
from typing import Any

from torch.utils.data import DataLoader, DistributedSampler


class BaseDataModule(abc.ABC):
    """
    Abstract base class for defining a DataModule.

    Map-style modules iterate a finite Dataset with DistributedSampler.
    Stream-style modules use IterableDataset and fixed steps_per_epoch.
    """

    is_streaming: bool = False

    def __init__(self, config: Any):
        self.config = config

    @abc.abstractmethod
    def prepare_data(self):
        """Download or prepare data. Only called once per node in DDP."""
        pass

    @abc.abstractmethod
    def setup(self, stage: str | None = None):
        """Load and split data. Called on every GPU."""
        pass

    @abc.abstractmethod
    def train_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler | None]:
        """Returns the DataLoader and optional DistributedSampler for training."""
        pass

    @abc.abstractmethod
    def val_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader | None, DistributedSampler | None]:
        """Returns the DataLoader and optional DistributedSampler for validation."""
        pass


class MapDataModule(BaseDataModule):
    """Finite dataset module using DistributedSampler during training."""

    is_streaming = False


class StreamDataModule(BaseDataModule):
    """Iterable dataset module for unbounded / large corpora."""

    is_streaming = True

    def validate_streaming_config(self) -> None:
        steps = getattr(self.config.data, "steps_per_epoch", None)
        if steps is None or steps <= 0:
            raise ValueError("Streaming DataModules require data.steps_per_epoch > 0.")
