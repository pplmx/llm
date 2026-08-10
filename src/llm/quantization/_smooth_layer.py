"""SmoothQuantLinear: weight+activation INT8 Linear with activation smoothing.

Storage convention:
- ``weight_packed``: int8 weights, shape [out_features * in_features]
  (SmoothQuant is an INT8 method; no nibble packing).
- ``weight_scales``: per-output-row fp16 scales [out_features] —
  ``w_int8 * weight_scales`` dequantizes the smoothed weights.
- ``act_scale``: per-tensor fp16 activation scale (max abs / 127).
- ``input_scales``: per-input-channel smoothing factors ``s`` [in_features]
  (fp16). The layer was quantized from ``W·s`` and compensates at forward
  time by dividing the input: ``y = Q8(W·s)·Q8(x/s)``.

Keeping the smoothing compensation in the layer (rather than folding it into
the preceding layer) is exact and needs no graph analysis; cross-layer
folding is a follow-up optimization (see ADR-010).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SmoothQuantLinear(nn.Module):
    """INT8 weight+activation quantized Linear with per-channel smoothing.

    Attributes:
        in_features / out_features: linear geometry.
        sym: must be True (asymmetric SmoothQuant is a follow-up).
        input_scales: per-input-channel smoothing scale ``s``, shape
            [in_features], or None when the caller already folded the
            scales upstream.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        weight_packed: torch.Tensor,
        weight_scales: torch.Tensor,
        act_scale: torch.Tensor,
        sym: bool = True,
        input_scales: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sym = sym

        self.register_buffer("weight_packed", weight_packed)
        self.register_buffer("weight_scales", weight_scales)
        self.register_buffer("act_scale", act_scale)
        if input_scales is not None:
            self.register_buffer("input_scales", input_scales)
        else:
            self.input_scales = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def _dequantize_weights(self) -> torch.Tensor:
        """Dequantize the smoothed int8 weights: [out_features, in_features] fp32."""
        weight_packed = self.weight_packed
        if not isinstance(weight_packed, torch.Tensor):
            raise RuntimeError("SmoothQuant packed weights were not initialized")
        weight_scales = self.weight_scales
        if not isinstance(weight_scales, torch.Tensor):
            raise RuntimeError("SmoothQuant weight scales were not initialized")
        w_int = weight_packed.reshape(self.out_features, self.in_features).to(torch.float32)
        return w_int * weight_scales.to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ``Q8(W·s)·Q8(x/s)`` with INT8 fake quantization.

        Activations are quantized per-tensor (the SmoothQuant contract), then
        multiplied by the dequantized smoothed weights.

        Args:
            x: Input tensor of shape [..., in_features].

        Returns:
            Output tensor of shape [..., out_features], dtype fp32.

        Raises:
            NotImplementedError: If ``sym=False`` was passed at construction.
        """
        if not self.sym:
            raise NotImplementedError("Asymmetric SmoothQuant forward is not yet implemented. Construct with sym=True.")

        if self.input_scales is not None:
            x = x / self.input_scales.to(x.dtype)

        # Per-tensor INT8 activation fake-quantization.
        act_scale = self.act_scale
        if not isinstance(act_scale, torch.Tensor):
            raise RuntimeError("SmoothQuant activation scale was not initialized")
        act_scale = act_scale.to(x.dtype)
        x_q = torch.clamp(torch.round(x / act_scale), -128, 127) * act_scale

        w_fp = self._dequantize_weights()
        # Weights are materialised in fp32; upcast the (fake-quantised) input
        # so fp16/bf16 model inference works.  Output is fp32.
        return torch.nn.functional.linear(x_q.to(torch.float32), w_fp, self.bias)
