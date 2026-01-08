"""
QLoRA (Quantized LoRA) Module.

Implements memory-efficient fine-tuning by combining:
1. 4-bit quantization of base model weights (NF4 format)
2. Low-rank adaptation (LoRA) in full precision

Reference: https://arxiv.org/abs/2305.14314
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# NF4 quantization constants (Normal Float 4-bit)
# Values are optimized for normally distributed weights
NF4_LEVELS = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
)


def quantize_nf4(weight: torch.Tensor, block_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weights to 4-bit NF4 format with block-wise scaling.

    Args:
        weight: Weight tensor to quantize [out_features, in_features]
        block_size: Number of elements per quantization block

    Returns:
        Tuple of (quantized_indices, scales)
        - quantized_indices: uint8 tensor with 4-bit indices packed
        - scales: Absmax scale per block
    """
    weight_flat = weight.flatten().float()
    n_elements = weight_flat.numel()

    # Pad to multiple of block_size
    n_blocks = (n_elements + block_size - 1) // block_size
    padded_size = n_blocks * block_size
    if padded_size > n_elements:
        weight_flat = F.pad(weight_flat, (0, padded_size - n_elements))

    # Reshape to blocks
    weight_blocks = weight_flat.view(n_blocks, block_size)

    # Compute absmax scale per block
    scales = weight_blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)

    # Normalize to [-1, 1]
    normalized = weight_blocks / scales

    # Quantize to nearest NF4 level
    nf4 = NF4_LEVELS.to(weight.device)
    distances = (normalized.unsqueeze(-1) - nf4.view(1, 1, -1)).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)

    # Store original shape info
    return indices.flatten()[:n_elements].view(weight.shape), scales.flatten()[:n_blocks]


def dequantize_nf4(
    indices: torch.Tensor,
    scales: torch.Tensor,
    original_shape: tuple[int, ...],
    block_size: int = 64,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Dequantize 4-bit NF4 indices back to floating point.

    Args:
        indices: Quantized indices (uint8)
        scales: Block-wise scales
        original_shape: Original weight shape
        block_size: Block size used during quantization
        dtype: Target dtype for dequantized weights

    Returns:
        Dequantized weight tensor
    """
    nf4 = NF4_LEVELS.to(indices.device)
    weight_flat = nf4[indices.flatten().long()]

    n_elements = weight_flat.numel()
    n_blocks = scales.numel()

    # Pad to match blocks
    padded_size = n_blocks * block_size
    if padded_size > n_elements:
        weight_flat = F.pad(weight_flat, (0, padded_size - n_elements))

    # Apply scales
    weight_blocks = weight_flat.view(n_blocks, block_size)
    weight_scaled = weight_blocks * scales.view(-1, 1)

    return weight_scaled.flatten()[:n_elements].view(original_shape).to(dtype)


class QLoRALinear(nn.Module):
    """
    QLoRA-adapted linear layer.

    Combines 4-bit quantized base weights with full-precision LoRA adapters.
    Memory usage: ~4x reduction for base weights.

    Args:
        base_layer: Original nn.Linear layer to adapt (will be quantized)
        rank: LoRA rank (default: 8)
        alpha: LoRA alpha (default: 16.0)
        dropout: Dropout for LoRA path (default: 0.0)
        block_size: Quantization block size (default: 64)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        block_size: int = 64,
    ):
        super().__init__()

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.block_size = block_size

        # Store original dtype for dequantization
        self.compute_dtype = base_layer.weight.dtype
        device = base_layer.weight.device

        # Quantize base weights to 4-bit NF4
        with torch.no_grad():
            indices, scales = quantize_nf4(base_layer.weight.detach(), block_size)
            self.register_buffer("weight_indices", indices)
            self.register_buffer("weight_scales", scales)
            self.original_shape = base_layer.weight.shape

        # Handle bias
        if base_layer.bias is not None:
            self.register_buffer("bias", base_layer.bias.detach().clone())
        else:
            self.bias = None

        # LoRA adapters in full precision
        self.lora_A = nn.Parameter(torch.empty(self.in_features, rank, device=device, dtype=self.compute_dtype))
        self.lora_B = nn.Parameter(torch.empty(rank, self.out_features, device=device, dtype=self.compute_dtype))

        # Dropout for regularization
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize LoRA weights
        self._init_lora_weights()

    def _init_lora_weights(self) -> None:
        """Initialize LoRA weights following standard LoRA initialization."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantized base + LoRA adaptation."""
        # Dequantize base weights on-the-fly
        weight = dequantize_nf4(
            self.weight_indices,
            self.weight_scales,
            self.original_shape,
            self.block_size,
            self.compute_dtype,
        )

        # Base layer output
        base_output = F.linear(x, weight, self.bias)

        # LoRA output
        lora_output = self.lora_dropout(x) @ self.lora_A @ self.lora_B

        return base_output + lora_output * self.scaling

    @property
    def trainable_parameters(self) -> int:
        """Number of trainable LoRA parameters."""
        return self.lora_A.numel() + self.lora_B.numel()

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, block_size={self.block_size}"
        )


def apply_qlora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    block_size: int = 64,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """
    Apply QLoRA to specified linear layers in a model.

    Args:
        model: The model to adapt
        rank: LoRA rank
        alpha: LoRA alpha (scaling = alpha / rank)
        dropout: Dropout probability for LoRA path
        block_size: Quantization block size
        target_modules: List of module name patterns to target.
                        If None, targets all nn.Linear layers.

    Returns:
        The model with QLoRA applied (modified in-place)
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
        qlora_layer = QLoRALinear(module, rank=rank, alpha=alpha, dropout=dropout, block_size=block_size)

        # Navigate to parent and replace
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], qlora_layer)

    return model


def get_qlora_parameters(model: nn.Module):
    """Get only QLoRA trainable parameters for optimizer."""
    for module in model.modules():
        if isinstance(module, QLoRALinear):
            yield module.lora_A
            yield module.lora_B
