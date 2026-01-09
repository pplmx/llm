"""
Calibration for Quantization.

Collects activation statistics for quantization scale computation.
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class ActivationStats:
    """Statistics for a single layer's activations."""

    name: str
    min_val: float = float("inf")
    max_val: float = float("-inf")
    abs_max: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    num_samples: int = 0

    def update(self, tensor: torch.Tensor) -> None:
        """Update statistics with new tensor."""
        tensor = tensor.detach().float()

        batch_min = tensor.min().item()
        batch_max = tensor.max().item()
        batch_abs_max = tensor.abs().max().item()
        batch_mean = tensor.mean().item()
        batch_std = tensor.std().item()
        batch_size = tensor.numel()

        # Running statistics
        total_samples = self.num_samples + batch_size

        # Update min/max
        self.min_val = min(self.min_val, batch_min)
        self.max_val = max(self.max_val, batch_max)
        self.abs_max = max(self.abs_max, batch_abs_max)

        # Welford's online algorithm for mean and variance
        old_mean = self.mean
        self.mean = old_mean + (batch_mean - old_mean) * batch_size / total_samples

        self.num_samples = total_samples

    def compute_scale(self, bits: int = 8, symmetric: bool = True) -> float:
        """Compute quantization scale."""
        qmax = 2 ** (bits - 1) - 1 if symmetric else 2**bits - 1

        if symmetric:
            scale = self.abs_max / qmax if self.abs_max > 0 else 1.0
        else:
            scale = (self.max_val - self.min_val) / qmax if self.max_val > self.min_val else 1.0

        return max(scale, 1e-8)


class CalibrationDataCollector:
    """
    Collects activation statistics for quantization calibration.

    Hooks into model forward passes to record min/max/mean/std
    of activations at each layer.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize collector.

        Args:
            model: Model to collect statistics from.
        """
        self.model = model
        self.stats: dict[str, ActivationStats] = {}
        self.hooks: list[Any] = []

    def register_hooks(self, layer_types: tuple = (nn.Linear,)) -> None:
        """
        Register forward hooks on specified layer types.

        Args:
            layer_types: Tuple of layer types to hook.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, layer_types):
                self.stats[name] = ActivationStats(name=name)
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

        logger.info(f"Registered {len(self.hooks)} calibration hooks")

    def _make_hook(self, name: str):
        """Create a forward hook for a named layer."""

        def hook(module, input, output):  # noqa: A002 - hook API uses 'input'
            if isinstance(output, tuple):
                output = output[0]
            self.stats[name].update(output)

        return hook

    def collect(
        self,
        dataloader: DataLoader,
        num_batches: int | None = None,
        device: str | torch.device = "cuda",
    ) -> dict[str, ActivationStats]:
        """
        Collect activation statistics from calibration data.

        Args:
            dataloader: Calibration data loader.
            num_batches: Maximum number of batches to process.
            device: Device to run on.

        Returns:
            Dictionary of layer name to activation stats.
        """
        self.model.eval()
        self.model.to(device)

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if num_batches and i >= num_batches:
                    break

                # Handle different batch formats
                if isinstance(batch, dict):
                    input_ids = batch.get("input_ids", batch.get("inputs"))
                elif isinstance(batch, (tuple, list)):
                    input_ids = batch[0]
                else:
                    input_ids = batch

                input_ids = input_ids.to(device)
                self.model(input_ids)

                if (i + 1) % 10 == 0:
                    logger.info(f"Calibrated {i + 1} batches")

        logger.info(f"Calibration complete: {sum(s.num_samples for s in self.stats.values())} samples")
        return self.stats

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_scales(self, bits: int = 8, symmetric: bool = True) -> dict[str, float]:
        """
        Compute quantization scales for all layers.

        Args:
            bits: Quantization bit width.
            symmetric: Whether to use symmetric quantization.

        Returns:
            Dictionary of layer name to scale.
        """
        return {name: stats.compute_scale(bits, symmetric) for name, stats in self.stats.items()}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()
