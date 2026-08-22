"""Fake quantization for Quantization-Aware Training (QAT).

ROADMAP 13.2 / RIL TASK-218, DEC-054: PTQ is already covered (simple /
GPTQ / AWQ / SmoothQuant / FP8). QAT keeps the *deployment* quantization in
the forward but flows a **straight-through estimator** (STE) gradient into
the underlying float parameters, so training adapts weights to the quantized
grid the model will later be packed into (typically recovering accuracy that
round-to-nearest PTQ loses).

This module provides a uniform, symmetric fake quantiser for INT8 / INT4
weights (per-channel dynamic scale) and, optionally, per-tensor activations,
with scale = absmax / qmax computed fresh each forward (dynamic scaling —
learned / calibration scales are the TASK-220 slice). ``round`` is steered:
forward rounds, backward passes the gradient through unchanged, so the
underlying fp32 ``Parameter`` never gets a zero gradient from the quantiser.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class _StraightThroughRound(torch.autograd.Function):
    """Round in the forward, identity gradient in the backward (STE)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x.round()

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        return grad_outputs[0]


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round ``x`` to the nearest integer with a straight-through gradient."""
    return _StraightThroughRound.apply(x)


def _expanded_scale(scale: torch.Tensor, channel_dim: int, x: torch.Tensor) -> torch.Tensor:
    """Reshape a (scalar or channel-length) scale to broadcast over ``x``."""
    if scale.numel() == 1:
        return scale
    shape = [1] * x.dim()
    shape[channel_dim] = x.shape[channel_dim]
    return scale.reshape(shape)


class FakeQuantize(nn.Module):
    """Uniform symmetric fake quantizer (dynamic scale, straight-through).

    ``q = clamp(round(x / scale), -qmax, qmax) * scale`` where
    ``scale = absmax / qmax`` is adaptive. ``per_channel`` quantizes along
    ``channel_dim`` (weights: output rows). Used for weights (per-channel) and
    activations (per-tensor).
    """

    def __init__(
        self,
        bits: int = 8,
        *,
        per_channel: bool = False,
        channel_dim: int = 0,
        floor_cap: float | None = None,
        static_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if bits not in (4, 8):
            raise ValueError(f"FakeQuantize supports 4 or 8 bits, got {bits}")
        self.bits = bits
        self.qmax = float((1 << (bits - 1)) - 1)
        self.per_channel = per_channel
        self.channel_dim = channel_dim
        self.floor_cap = floor_cap
        if static_scale is not None:
            self.register_buffer("static_scale", static_scale.detach())
        else:
            self.static_scale = None

    def forward(self, x: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
        if self.floor_cap is not None:
            x = x.clamp(min=self.floor_cap)
        if scale is None:
            if self.static_scale is not None:
                scale = _expanded_scale(self.static_scale, self.channel_dim, x)
            elif self.per_channel:
                base = x.detach().abs()
                reduce_dims = tuple(i for i in range(x.dim()) if i != self.channel_dim)
                amax = base.amax(dim=reduce_dims, keepdim=True)
                scale = (amax / self.qmax).clamp_min(1e-8)
            else:
                scale = x.detach().abs().max().clamp_min(1e-8) / self.qmax
        else:
            scale = _expanded_scale(scale, self.channel_dim, x)
        q = ste_round(x / scale).clamp(-self.qmax, self.qmax)
        return q * scale


class FakeQuantLinear(nn.Module):
    """An ``nn.Linear`` whose forward fake-quantizes weights (+ activations).

    Keeps the full-precision ``weight``/``bias`` as trainable Parameters and
    applies :class:`FakeQuantize` (dynamic scale, STE) to the weights every
    forward (per-channel) and, if ``quant_activation``, to the input
    (per-tensor). Backprop flows through the quantizer (STE) so the fp32
    weights are updated by the optimizer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        bits: int = 8,
        quant_activation: bool = False,
        floor_cap: float | None = None,
        weight_scale: torch.Tensor | None = None,
        activation_scale: torch.Tensor | None = None,
        learnable_scales: bool = False,
    ) -> None:
        super().__init__()
        if bits not in (4, 8):
            raise ValueError(f"FakeQuantLinear supports 4 or 8 bits, got {bits}")
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.quant_activation = quant_activation
        self.floor_cap = floor_cap
        self.learnable_scales = learnable_scales
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.fake_w = FakeQuantize(bits, per_channel=True, channel_dim=0)
        self.fake_a = FakeQuantize(bits, per_channel=False, floor_cap=floor_cap) if quant_activation else None
        self._register_static_scale("weight_scale_param", weight_scale, per_channel=True)
        self._register_static_scale("activation_scale_param", activation_scale, per_channel=False)
        self.reset_parameters()

    def _register_static_scale(self, attr: str, scale: torch.Tensor | None, *, per_channel: bool) -> None:
        if scale is None:
            setattr(self, attr, None)
            return
        s = scale.detach().float()
        if self.learnable_scales:
            self.register_parameter(attr, nn.Parameter(s))
        else:
            self.register_buffer(attr, s)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fake_a is not None:
            x = self.fake_a(x, scale=self.activation_scale_param)
        w = self.fake_w(self.weight, scale=self.weight_scale_param)
        return functional.linear(x, w, self.bias)


def _module_matches(name: str, target_modules) -> bool:
    if not target_modules:
        return True
    return any(name.endswith(t) for t in target_modules)


def apply_fake_quant(
    model: nn.Module,
    *,
    bits: int = 8,
    quant_activation: bool = False,
    target_modules=None,
) -> nn.Module:
    """Replace matching ``nn.Linear`` layers with :class:`FakeQuantLinear`.

    The replacement copies the source weights/biases and keeps them as
    full-precision trainable Parameters (fake quantization only rounds in the
    forward via the STE), so the QAT training loop can adapt them. ``target_modules``
    is an iterable of name suffixes (e.g. ``("fc1", "fc2", "qkv_proj")``) or
    ``None``/empty to quantize every ``nn.Linear``.
    """
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    targets = [t for t in targets if _module_matches(t[0], target_modules)]
    if not targets:
        raise ValueError(f"apply_fake_quant matched no nn.Linear layers for target_modules={target_modules!r}.")
    for name, module in targets:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        replacement = FakeQuantLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            bits=bits,
            quant_activation=quant_activation,
        )
        with torch.no_grad():
            replacement.weight.copy_(module.weight)
            if module.bias is not None and replacement.bias is not None:
                replacement.bias.copy_(module.bias)
        setattr(parent, parts[-1], replacement)
    return model
