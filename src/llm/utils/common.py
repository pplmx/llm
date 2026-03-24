import math
from collections.abc import Callable

import torch
from torch import nn


def make_factory_kwargs(
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.device | torch.dtype]:
    """
    Create factory_kwargs dict for nn.Module constructors.

    Args:
        device: Target device.
        dtype: Target data type.

    Returns:
        Dict with device and dtype keys (only non-None values).
    """
    kwargs: dict[str, torch.device | torch.dtype] = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype
    return kwargs


def get_activation_layer(activation_name: str) -> Callable:
    """
    根据名称获取激活函数层.

    Args:
        activation_name: 激活函数名称.

    Returns:
        对应的激活函数类.

    Raises:
        ValueError: 当提供的激活函数名称不支持时.
    """
    match activation_name.lower():
        case "gelu":
            return nn.GELU
        case "relu":
            return nn.ReLU
        case "silu" | "swish":  # swish is SiLU
            return nn.SiLU
        case "glu":
            # Gated Linear Unit
            return nn.GLU
        case "leaky_relu":
            return nn.LeakyReLU
        case _:
            supported_activations = ["gelu", "relu", "silu", "swish", "glu", "leaky_relu"]
            raise ValueError(
                f"Unsupported activation function: {activation_name}. Supported activations are: {supported_activations}"
            )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Counts the total and trainable parameters of a PyTorch model.

    Args:
        model (nn.Module): The model to inspect.

    Returns:
        tuple[int, int]: A tuple containing:
            - total_params (int): The total number of parameters.
            - trainable_params (int): The number of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def init_lora_weights(lora_A: nn.Parameter, lora_B: nn.Parameter) -> None:
    """
    Initialize LoRA weights: A with Kaiming uniform, B with zeros.

    This ensures the LoRA adaptation starts at zero (output = base_output initially).

    Args:
        lora_A: LoRA A matrix (in_features x rank)
        lora_B: LoRA B matrix (rank x out_features)
    """
    nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
    nn.init.zeros_(lora_B)
