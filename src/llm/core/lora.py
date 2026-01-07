"""
LoRA (Low-Rank Adaptation) Module.

Implements parameter-efficient fine-tuning by adding trainable low-rank
matrices to frozen linear layers.

Reference: https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA-adapted linear layer.

    Wraps a frozen nn.Linear and adds trainable low-rank matrices A and B.
    Output: base_output + (input @ A) @ B * scaling

    Args:
        base_layer: The original nn.Linear layer to adapt (will be frozen)
        rank: Rank of the low-rank matrices (default: 8)
        alpha: Scaling factor (default: 16.0)
        dropout: Dropout probability for LoRA path (default: 0.0)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # LoRA matrices (same device/dtype as base)
        self.lora_A = nn.Parameter(torch.empty(in_features, rank, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(rank, out_features, device=device, dtype=dtype))

        # Dropout for regularization
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize
        self._init_lora_weights()

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def _init_lora_weights(self) -> None:
        """Initialize LoRA weights: A with Kaiming, B with zeros."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: frozen base + LoRA adaptation."""
        base_output = self.base_layer(x)
        lora_output = self.lora_dropout(x) @ self.lora_A @ self.lora_B
        return base_output + lora_output * self.scaling

    def merge_weights(self) -> None:
        """Merge LoRA weights into the base layer for efficient inference."""
        with torch.no_grad():
            delta_w = (self.lora_A @ self.lora_B) * self.scaling
            self.base_layer.weight.add_(delta_w.T)

    def unmerge_weights(self) -> None:
        """Unmerge LoRA weights from the base layer."""
        with torch.no_grad():
            delta_w = (self.lora_A @ self.lora_B) * self.scaling
            self.base_layer.weight.sub_(delta_w.T)

    @property
    def trainable_parameters(self) -> int:
        """Number of trainable LoRA parameters."""
        return self.lora_A.numel() + self.lora_B.numel()

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """
    Apply LoRA to specified linear layers in a model.

    Args:
        model: The model to adapt
        rank: LoRA rank
        alpha: LoRA alpha (scaling = alpha / rank)
        dropout: Dropout probability for LoRA path
        target_modules: List of module name patterns to target.
                        If None, targets all nn.Linear layers.

    Returns:
        The model with LoRA applied (modified in-place)
    """
    if target_modules is None:
        target_modules = []

    def should_apply(name: str) -> bool:
        if not target_modules:
            return True
        return any(pattern in name for pattern in target_modules)

    # Collect modules to replace
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_apply(name):
            replacements.append((name, module))

    # Apply replacements
    for name, module in replacements:
        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)

        # Navigate to parent and replace
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora_layer)

    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights into base layers for efficient inference.

    Args:
        model: Model with LoRA layers

    Returns:
        The model with merged weights (modified in-place)
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
    return model


def unmerge_lora(model: nn.Module) -> nn.Module:
    """
    Unmerge all LoRA weights from base layers.

    Args:
        model: Model with merged LoRA layers

    Returns:
        The model with unmerged weights (modified in-place)
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge_weights()
    return model


def get_lora_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """
    Get only LoRA parameters for optimizer.

    Args:
        model: Model with LoRA layers

    Yields:
        LoRA parameters (lora_A and lora_B)
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A
            yield module.lora_B


def count_lora_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Count trainable and total parameters in a model with LoRA.

    Args:
        model: Model with LoRA layers

    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def disable_lora(model: nn.Module) -> None:
    """Disable LoRA by setting scaling to 0."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module._original_scaling = module.scaling
            module.scaling = 0.0


def enable_lora(model: nn.Module) -> None:
    """Re-enable LoRA after disabling."""
    for module in model.modules():
        if isinstance(module, LoRALinear) and hasattr(module, "_original_scaling"):
            module.scaling = module._original_scaling
