from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    """Base class for all evaluation metrics."""

    name: str

    @abstractmethod
    def compute(self, predictions: Any, references: Any) -> dict:
        """Compute metric score.

        Args:
            predictions: Model outputs
            references: Ground truth

        Returns:
            Dictionary with metric name and score
        """
        pass
