"""
Post-Training Quantization (PTQ).

Provides utilities for quantizing models after training.
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class QuantConfig:
    """Configuration for quantization."""

    bits: int = 8
    symmetric: bool = True
    per_channel: bool = False
    dynamic: bool = False  # Dynamic vs static quantization

    def __post_init__(self):
        if self.bits not in (4, 8):
            raise ValueError(f"Unsupported bit width: {self.bits}. Use 4 or 8.")


class QuantizedLinear(nn.Module):
    """
    Quantized Linear layer with INT8/INT4 weights.

    Stores quantized weights and scales, dequantizes during forward.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: QuantConfig | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantConfig()

        # Quantized weights (stored as int8)
        self.register_buffer(
            "weight_quantized",
            torch.zeros(out_features, in_features, dtype=torch.int8),
        )

        # Scales for dequantization
        if self.config.per_channel:
            self.register_buffer("weight_scale", torch.ones(out_features))
        else:
            self.register_buffer("weight_scale", torch.ones(1))

        # Zero point for asymmetric quantization
        if not self.config.symmetric:
            self.register_buffer("weight_zero_point", torch.zeros_like(self.weight_scale))
        else:
            self.weight_zero_point = None

        # Bias remains in fp32
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantized weights."""
        # Dequantize weights
        weight = self._dequantize_weight()

        return nn.functional.linear(x, weight, self.bias)

    def _dequantize_weight(self) -> torch.Tensor:
        """Dequantize stored weights."""
        weight = self.weight_quantized.float()

        if self.weight_zero_point is not None:
            weight = weight - self.weight_zero_point.view(-1, 1)

        weight = weight * self.weight_scale.view(-1, 1)

        return weight

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        config: QuantConfig | None = None,
        scale: float | None = None,
    ) -> "QuantizedLinear":
        """
        Create QuantizedLinear from a regular Linear layer.

        Args:
            linear: Source Linear layer.
            config: Quantization configuration.
            scale: Pre-computed scale (optional).

        Returns:
            Quantized layer.
        """
        config = config or QuantConfig()
        quant_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            config=config,
        )

        # Quantize weights
        weight = linear.weight.data

        if config.per_channel:
            # Per-channel quantization
            if scale is None:
                abs_max = weight.abs().max(dim=1)[0]
                qmax = 2 ** (config.bits - 1) - 1
                scale = abs_max / qmax
                scale = scale.clamp(min=1e-8)

            weight_quantized = (weight / scale.view(-1, 1)).round().clamp(-128, 127).to(torch.int8)
            quant_linear.weight_scale.copy_(scale)
        else:
            # Per-tensor quantization
            if scale is None:
                abs_max = weight.abs().max()
                qmax = 2 ** (config.bits - 1) - 1
                scale = abs_max / qmax
                scale = max(scale.item(), 1e-8)

            weight_quantized = (weight / scale).round().clamp(-128, 127).to(torch.int8)
            quant_linear.weight_scale.fill_(scale)

        quant_linear.weight_quantized.copy_(weight_quantized)

        if linear.bias is not None:
            quant_linear.bias.data.copy_(linear.bias.data)

        return quant_linear


def quantize_linear_layer(
    layer: nn.Linear,
    config: QuantConfig | None = None,
    scale: float | None = None,
) -> QuantizedLinear:
    """
    Quantize a single Linear layer.

    Args:
        layer: Linear layer to quantize.
        config: Quantization configuration.
        scale: Pre-computed scale.

    Returns:
        Quantized layer.
    """
    return QuantizedLinear.from_linear(layer, config, scale)


def quantize_model(
    model: nn.Module,
    config: QuantConfig | None = None,
    scales: dict[str, float] | None = None,
    inplace: bool = False,
) -> nn.Module:
    """
    Quantize all Linear layers in a model.

    Args:
        model: Model to quantize.
        config: Quantization configuration.
        scales: Pre-computed scales per layer name.
        inplace: Whether to modify model in-place.

    Returns:
        Quantized model.
    """
    config = config or QuantConfig()
    scales = scales or {}

    if not inplace:
        import copy

        model = copy.deepcopy(model)

    # Track replacements
    replacements = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            scale = scales.get(name)
            quant_layer = QuantizedLinear.from_linear(module, config, scale)
            replacements.append((name, quant_layer))

    # Apply replacements
    for name, quant_layer in replacements:
        _replace_module(model, name, quant_layer)

    logger.info(f"Quantized {len(replacements)} linear layers")

    return model


def _replace_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a module by name."""
    parts = name.split(".")
    parent = model

    for part in parts[:-1]:
        parent = getattr(parent, part)

    setattr(parent, parts[-1], new_module)


def compute_model_size(model: nn.Module) -> dict[str, Any]:
    """
    Compute model size statistics.

    Returns:
        Dictionary with size information.
    """
    total_params = 0
    total_bytes = 0
    quantized_layers = 0

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            quantized_layers += 1
            # INT8 weights
            total_params += module.weight_quantized.numel()
            total_bytes += module.weight_quantized.numel()  # 1 byte per int8
            # FP32 scales
            total_bytes += module.weight_scale.numel() * 4
            if module.bias is not None:
                total_bytes += module.bias.numel() * 4
        elif isinstance(module, nn.Linear):
            total_params += module.weight.numel()
            total_bytes += module.weight.numel() * module.weight.element_size()
            if module.bias is not None:
                total_bytes += module.bias.numel() * module.bias.element_size()

    return {
        "total_params": total_params,
        "total_bytes": total_bytes,
        "size_mb": total_bytes / (1024 * 1024),
        "quantized_layers": quantized_layers,
    }
