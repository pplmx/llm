"""
GPTQQuantizedLinear: GPTQ-quantized Linear with packed 4-bit (or 8-bit) storage.

Storage convention for bits=4:
- weight_packed: int8 tensor, two int4 values per byte.
  Pair (w[2i], w[2i+1]) packed as (w[2i] << 4) | (w[2i+1] & 0x0F).
- scales: float16 tensor, shape [out_features, in_features // group_size].
- zeros: int8 tensor (or None if sym=True), shape [out_features, in_features // group_size].
- group_size=-1: scales shape [out_features, 1] (per-channel).
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _pack_4bit(w: torch.Tensor) -> torch.Tensor:
    """Pack unsigned int4 values (shape [N,], even N) into int8 storage.

    Each pair (w[2i], w[2i+1]) is stored as (w[2i] << 4) | (w[2i+1] & 0x0F).

    Args:
        w: int8 tensor of shape [N,] with values in [0, 15]. N must be even.

    Returns:
        int8 tensor of shape [N // 2,].

    Raises:
        ValueError: If N is odd or values are out of [0, 15].
    """
    if w.numel() % 2 != 0:
        raise ValueError(f"_pack_4bit requires even number of values, got {w.numel()}.")
    if w.min() < 0 or w.max() > 15:
        raise ValueError(f"_pack_4bit values must be in [0, 15], got range [{w.min().item()}, {w.max().item()}].")

    w_even = w[0::2]
    w_odd = w[1::2]
    packed = ((w_even << 4) | (w_odd & 0x0F)).to(torch.int8)
    return packed


def _unpack_4bit(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Unpack int8 storage back to unsigned int4 values of shape [numel]."""
    if numel % 2 != 0:
        raise ValueError(f"_unpack_4bit numel must be even, got {numel}.")

    # Flatten to 1D so callers can pass either flat or pre-shaped packed tensors.
    packed_flat = packed.reshape(-1)

    # High nibble: even indices, Low nibble: odd indices
    high = (packed_flat >> 4) & 0x0F
    low = packed_flat & 0x0F

    out = torch.zeros(numel, dtype=torch.int8, device=packed.device)
    out[0::2] = high
    out[1::2] = low
    return out


class GPTQQuantizedLinear(nn.Module):
    """GPTQ-quantized Linear with packed 4-bit (or 8-bit) weight storage."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        weight_packed: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor | None,
        bits: int = 4,
        group_size: int = 128,
        sym: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.sym = sym

        # Register packed weights and scales as buffers (not Parameters — no grad)
        self.register_buffer("weight_packed", weight_packed)
        self.register_buffer("scales", scales)
        if zeros is not None:
            self.register_buffer("zeros", zeros)
        else:
            self.zeros = None

        # Bias remains fp32 / Parameter (only if original layer had bias)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def _unpack_weights(self) -> torch.Tensor:
        """Unpack int8 storage to int4 (or int8) tensor of shape [out_features, in_features]."""
        if self.bits == 4:
            unpacked = _unpack_4bit(self.weight_packed, numel=self.out_features * self.in_features)
            return unpacked.reshape(self.out_features, self.in_features)
        else:
            # 8-bit: weight_packed stores int8 values directly
            return self.weight_packed.reshape(self.out_features, self.in_features)
