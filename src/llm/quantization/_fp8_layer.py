"""Fp8QuantizedLinear: FP8 (E4M3/E5M2) weight + activation quantization.

Stores the weights as a REAL ``torch.float8_e4m3fn`` (or ``e5m2``) buffer —
1 byte per weight, a genuine memory saving vs the INT8 ``QuantizedLinear``
(which stores int8) and a 4x saving over fp32 (2x over fp16). The forward
"simulated": the fp8 matmul itself is unavailable outside Hopper/Blackwell
hardware, so weights/activations are cast to fp8 and the product is computed
in fp32 (standard FP8 PTQ emulation) — numerically equivalent to what the
fp8 hardware paths of the larger frameworks produce per-tensor.

Scaling follows the common tensor-core convention: to use the full dynamic
range, ``x_q = round(x / scale)`` with ``scale = absmax / FP8_MAX``, stored
as a per-tensor (or per-output-row for weights) fp32 scale. Activation
scaling is either STATIC (``activation_scale`` captured at calibration time)
or DYNAMIC (absmax computed per forward).

The forward returns in the layer's effective dtype (RIL ISS-191 pattern) and
never mixes dtypes inside ``F.linear``, so a post-quant fp16/bf16 model cast
stays valid. True fp8 matmul kernels, fp8 KV cache, and fp8 backprop are out
of scope for this milestone.
"""

from __future__ import annotations

import torch
import torch.nn as nn

FP8_TYPES: dict[str, torch.dtype] = {
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
}
#: Maximum finite value of each FP8 format (saturation point).
FP8_MAX: dict[str, float] = {
    "e4m3": 448.0,
    "e5m2": 57344.0,
}

ENABLED = ("e4m3", "e5m2")


class Fp8QuantizedLinear(nn.Module):
    """FP8 weight(+activation) Linear with simulated (fp32) compute.

    Attributes:
        weight_fp8: the quantized weights, stored as an actual float8 buffer
            (shape ``[out_features, in_features]``, 1 byte/weight).
        weight_scale: fp32 dequantization scale — a scalar (per-tensor) or a
            per-output-row vector (per-channel).
        activation_scale: fp32 per-tensor activation scale captured at
            calibration, or ``None`` for dynamic (computed per forward).
        dtype_name: ``"e4m3"`` or ``"e5m2"`` (E4M3FN / E5M2).
        per_channel: whether ``weight_scale`` is per-output-row.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor | None,
        dtype_name: str = "e4m3",
        per_channel: bool = False,
    ):
        super().__init__()
        if dtype_name not in ENABLED:
            raise ValueError(f"Unsupported FP8 dtype {dtype_name!r}; expected one of {ENABLED}.")
        self.in_features = in_features
        self.out_features = out_features
        self.dtype_name = dtype_name
        self.per_channel = per_channel

        self.weight_fp8: torch.Tensor
        self.weight_scale: torch.Tensor
        self.activation_scale: torch.Tensor | None
        self.register_buffer("weight_fp8", weight_fp8)  # real float8 storage
        self.register_buffer("weight_scale", weight_scale)
        if activation_scale is not None:
            self.register_buffer("activation_scale", activation_scale)
        else:
            self.activation_scale = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    def _dequantize_weights(self) -> torch.Tensor:
        w_fp32 = self.weight_fp8.float()
        scale = self.weight_scale
        if scale.ndim > 0:
            scale = scale.view(-1, 1)
        return w_fp32 * scale.to(torch.float32)

    def _quantize_activations(self, x: torch.Tensor) -> torch.Tensor:
        """FP8 fake-quantize activations (static or dynamic per-tensor)."""
        fp8_dtype = FP8_TYPES[self.dtype_name]
        if self.activation_scale is not None:
            scale = self.activation_scale.to(x.dtype)
        else:
            scale = x.abs().max().clamp(min=1e-8) / FP8_MAX[self.dtype_name]
        x_scaled = x / scale
        # ``.to(fp8)`` saturates at the format max (448 / 57344) and rounds
        # to nearest-even — the tensor-core cast semantics.
        return x_scaled.to(fp8_dtype).to(x.dtype) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self.bias.dtype if self.bias is not None else x.dtype
        x_q = self._quantize_activations(x.to(torch.float32))
        w_fp32 = self._dequantize_weights()
        out = nn.functional.linear(
            x_q,
            w_fp32,
            self.bias.to(torch.float32) if self.bias is not None else self.bias,
        )
        return out.to(dtype)


def quantize_fp8_linear(
    layer: nn.Linear,
    dtype_name: str = "e4m3",
    per_channel: bool = True,
    activation_scale: torch.Tensor | None = None,
) -> Fp8QuantizedLinear:
    """Build an ``Fp8QuantizedLinear`` from an ``nn.Linear``.

    Weights are cast to the chosen FP8 format with ``scale = absmax /
    FP8_MAX`` (per-tensor or per-output-row). ``activation_scale`` may be a
    captured static per-tensor scale, or ``None`` for dynamic activation
    scaling at forward time.
    """
    if dtype_name not in ENABLED:
        raise ValueError(f"Unsupported FP8 dtype {dtype_name!r}; expected one of {ENABLED}.")
    fp8_dtype = FP8_TYPES[dtype_name]
    fmax = FP8_MAX[dtype_name]
    w = layer.weight.data
    if per_channel:
        scale = (w.abs().max(dim=1, keepdim=True)[0] / fmax).clamp(min=1e-8)  # [out,1]
        w_fp8 = (w / scale).to(fp8_dtype)
        scale = scale.squeeze(1)
    else:
        scale = torch.tensor(max(w.abs().max().item() / fmax, 1e-8), dtype=w.dtype, device=w.device)
        w_fp8 = (w / scale).to(fp8_dtype)
    # Keep the (possibly half) bias — Fp8QuantizedLinear.forward casts it.
    ql = Fp8QuantizedLinear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=layer.bias is not None,
        weight_fp8=w_fp8,
        weight_scale=scale.to(torch.float32),
        activation_scale=activation_scale,
        dtype_name=dtype_name,
        per_channel=per_channel,
    )
    if layer.bias is not None:
        ql.bias.data.copy_(layer.bias.data)
    ql.to(layer.weight.device)
    return ql
