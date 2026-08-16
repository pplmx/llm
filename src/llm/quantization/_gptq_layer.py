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
        weight_packed = self.weight_packed
        if not isinstance(weight_packed, torch.Tensor):
            raise RuntimeError("GPTQ packed weights were not initialized")
        if self.bits == 4:
            unpacked = _unpack_4bit(weight_packed, numel=self.out_features * self.in_features)
            return unpacked.reshape(self.out_features, self.in_features)
        else:
            # 8-bit: weight_packed stores int8 values directly
            return weight_packed.reshape(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantized weights.

        Dequantizes to fp32 and computes the matmul in fp32 for accuracy,
        then returns in the input's dtype — native ``nn.Linear`` semantics,
        keeping the layer a faithful drop-in replacement even after the
        model is cast to fp16/bf16 (serving default) or selectively
        quantized over a half base (RIL ISS-191). Asymmetric quantization
        (`sym=False`) is not yet implemented — raises NotImplementedError.

        Args:
            x: Input tensor of shape [..., in_features].

        Returns:
            Output tensor of shape [..., out_features].

        Raises:
            NotImplementedError: If `sym=False` was passed at construction.
        """
        if not self.sym:
            raise NotImplementedError("Asymmetric GPTQ forward is not yet implemented. Construct with sym=True.")

        scales = self.scales
        if not isinstance(scales, torch.Tensor):
            raise RuntimeError("GPTQ scales were not initialized")

        # Always dequantize from int4/int8 storage: trades compute for memory.
        # (Caching fp32 weights would double storage; deferred to future optimization.)
        w_int = self._unpack_weights()  # [out_features, in_features]
        # 4-bit storage is unsigned [0, 15] → shift to signed [-8, 7];
        # 8-bit storage is already signed int8 [-128, 127] (no shift).
        w_int_signed = w_int.to(torch.float32) - 8.0 if self.bits == 4 else w_int.to(torch.float32)

        if self.group_size == -1:
            # Per-channel: scales shape [out_features, 1] broadcasts across input dim.
            w_fp = w_int_signed * scales.to(torch.float32)
        else:
            # Per-group: scales shape [out_features, in_features // group_size].
            # Expand to [out_features, in_features] by repeating within each group.
            gs = self.group_size
            scales_expanded = scales.to(torch.float32).repeat_interleave(gs, dim=1)
            w_fp = w_int_signed * scales_expanded

        # Dequantize in fp32 for accuracy, then compute in fp32 and return
        # in the layer's effective dtype (RIL ISS-191). A post-quant
        # fp16/bf16 cast — the serving engine's
        # ``model.to(device, dtype=torch.float16)``, or selective
        # quantization over an already-half base — converts ``bias``/scales
        # to half; computing in fp32 avoids mixing dtypes inside
        # ``F.linear`` (crash) and returning the layer's dtype avoids
        # emitting fp32 into half-precision residual linears (crash). An
        # fp32 model is unchanged (fp32 in -> fp32 out).
        dtype = self.bias.dtype if self.bias is not None else x.dtype
        out = torch.nn.functional.linear(
            x.to(torch.float32),
            w_fp,
            self.bias.to(torch.float32) if self.bias is not None else self.bias,
        )
        return out.to(dtype)
