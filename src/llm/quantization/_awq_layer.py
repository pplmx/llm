"""AWQQuantizedLinear: activation-aware quantized Linear with packed storage.

Storage convention mirrors :class:`GPTQQuantizedLinear` (symmetric INT4/INT8
group quantization with per-group scales), plus one AWQ-specific buffer:

- ``input_scales``: per-input-channel scaling factors ``s`` of shape
  ``[in_features]`` (fp16). The layer was quantized from ``W * s`` and
  compensates at forward time by dividing the input: ``y = Q(W·s)·(x/s)``.

Keeping the compensation in the layer (rather than folding it into the
preceding layer) is exact and needs no graph analysis; cross-layer folding
is a follow-up optimization (see ADR-009).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from llm.quantization._gptq_layer import _unpack_4bit


class AWQQuantizedLinear(nn.Module):
    """Activation-aware weight-quantized Linear (symmetric group quantization).

    Attributes:
        in_features / out_features: linear geometry.
        bits: 4 or 8.
        group_size: -1 (per-channel) or positive int (per-group).
        sym: must be True (asymmetric AWQ is a follow-up).
        input_scales: per-input-channel AWQ scale ``s``, shape [in_features],
            or None when the caller already folded the scales upstream.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        weight_packed: torch.Tensor,
        scales: torch.Tensor,
        bits: int = 4,
        group_size: int = 128,
        sym: bool = True,
        input_scales: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.sym = sym

        self.register_buffer("weight_packed", weight_packed)
        self.register_buffer("scales", scales)
        if input_scales is not None:
            self.register_buffer("input_scales", input_scales)
        else:
            self.input_scales = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def _unpack_weights(self) -> torch.Tensor:
        """Unpack int8 storage to int4 (or int8) tensor [out_features, in_features]."""
        weight_packed = self.weight_packed
        if not isinstance(weight_packed, torch.Tensor):
            raise RuntimeError("AWQ packed weights were not initialized")
        if self.bits == 4:
            unpacked = _unpack_4bit(weight_packed, numel=self.out_features * self.in_features)
            return unpacked.reshape(self.out_features, self.in_features)
        return weight_packed.reshape(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ``Q(W·s)·(x/s)`` with exact AWQ scale compensation.

        Args:
            x: Input tensor of shape [..., in_features].

        Returns:
            Output tensor of shape [..., out_features], dtype fp32.

        Raises:
            NotImplementedError: If ``sym=False`` was passed at construction.
        """
        if not self.sym:
            raise NotImplementedError("Asymmetric AWQ forward is not yet implemented. Construct with sym=True.")

        if self.input_scales is not None:
            x = x / self.input_scales.to(x.dtype)

        w_int = self._unpack_weights()
        # 4-bit storage is unsigned [0, 15] → shift to signed [-8, 7];
        # 8-bit storage is already signed int8 [-128, 127] (no shift).
        w_int_signed = w_int.to(torch.float32) - 8.0 if self.bits == 4 else w_int.to(torch.float32)

        if self.group_size == -1:
            w_fp = w_int_signed * self.scales.to(torch.float32)
        else:
            gs = self.group_size
            scales_expanded = self.scales.to(torch.float32).repeat_interleave(gs, dim=1)
            w_fp = w_int_signed * scales_expanded

        return torch.nn.functional.linear(x, w_fp, self.bias)
