from abc import ABC, abstractmethod
from typing import Any


class BaseTask(ABC):
    """Base class for all evaluation tasks."""

    name: str
    metrics: list[Any]

    @abstractmethod
    def prepare_data(self, split: str) -> tuple[list[str], list[str]]:
        """Prepare inputs and references for evaluation.

        Args:
            split: Data split (e.g., 'train', 'test')

        Returns:
            Tuple of (inputs, references)
        """
        pass

    @abstractmethod
    def predict(self, model: Any, inputs: list[str]) -> list[str]:
        """Run model on inputs to get predictions.

        Args:
            model: Model to use for prediction
            inputs: List of input texts

        Returns:
            List of predicted outputs
        """
        pass
