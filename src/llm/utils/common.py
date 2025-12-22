from collections.abc import Callable

from torch import nn


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
