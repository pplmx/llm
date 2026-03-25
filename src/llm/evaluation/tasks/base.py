from abc import ABC, abstractmethod
from typing import Any


class BaseTask(ABC):
    """Base class for evaluation tasks."""

    name: str
    metrics: list[Any]

    @abstractmethod
    def prepare_data(self, split: str) -> tuple[list[str], list[str]]:
        """Prepare evaluation data."""
        pass

    @abstractmethod
    def predict(self, model: Any, inputs: list[str]) -> list[str]:
        """Generate predictions for inputs."""
        pass
