"""
Post-Training Quantization (PTQ).

Provides utilities for quantizing models after training.
Supports symmetric (scale-only) and asymmetric (scale + zero-point) 8-bit
weight quantization, per-tensor or per-channel. The asymmetric path stores
``q - 128`` in the int8 buffer and folds the offset into ``weight_zero_point``
so dequantization stays ``(q - zp) * scale``; it is exact on the grid and beats
symmetric on skewed (all-positive / all-negative) weight distributions.
"""

import logging
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class QuantConfig:
    """Configuration for quantization."""

    bits: int = 8
    symmetric: bool = True  # False = asymmetric (scale + zero-point) 8-bit
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

        # Type annotations for registered buffers (prevents ty from inferring
        # the broader `Tensor | Module` union that register_buffer creates).
        self.weight_quantized: torch.Tensor
        self.weight_scale: torch.Tensor

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
        self.weight_zero_point: torch.Tensor | None
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
        """Forward pass with dequantized weights.

        Dequantizes to fp32, computes in fp32 for accuracy, and returns in
        the layer's effective dtype (native ``nn.Linear`` semantics).  This
        keeps the layer a faithful drop-in even after the model is cast to
        fp16/bf16 — the serving engine's ``model.to(device, dtype=fp16)`` or
        selective quantization over a half base converts ``bias`` to half,
        and passing it straight into ``F.linear`` against fp32 weights
        crashed with a dtype mismatch (RIL TASK-196 / ISS-236, quant
        deep-dive F1; same fix as the GPTQ/AWQ/SmoothQuant layers in
        ISS-191).
        """
        weight = self._dequantize_weight()
        dtype = self.bias.dtype if self.bias is not None else x.dtype
        out = nn.functional.linear(
            x.to(torch.float32),
            weight,
            self.bias.to(torch.float32) if self.bias is not None else self.bias,
        )
        return out.to(dtype)

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
        scale: float | torch.Tensor | None = None,
    ) -> QuantizedLinear:
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
        if config.bits != 8:
            # The simple-PTQ layer stores weights in an int8 buffer. A
            # ``bits=4`` config quantized to a 4-bit *range* but persisted
            # them as full int8 values — no packing, no memory saving, while
            # still advertising a 4-bit model (RIL ISS-197). Real 4-bit
            # packing lives in the GPTQ/AWQ layers; fail fast instead of
            # silently storing 8-bit weights behind a 4-bit claim.
            raise NotImplementedError(
                f"Simple-PTQ only supports bits=8 (got bits={config.bits}). "
                "4-bit simple-PTQ would store the 4-bit grid as int8 with no "
                "memory saving; use the GPTQ or AWQ path "
                "(llm.quantization.gptq / llm.quantization.awq) for packed 4-bit."
            )
        quant_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            config=config,
        )

        # Quantize weights
        weight = linear.weight.data

        if not config.symmetric:
            # Asymmetric 8-bit weight quantization (scale + zero-point).
            # The 8-bit unsigned grid is [0, 255]; we store ``q - 128`` in the
            # int8 buffer and fold the offset into ``weight_zero_point`` so
            # dequantization still reads ``(q - zp) * scale`` (QuantizedLinear
            # already subtracts the zero point before scaling).
            if quant_linear.weight_zero_point is None:
                raise ValueError("asymmetric layer must have a zero-point buffer")
            qmax = (1 << config.bits) - 1  # 255
            if config.per_channel:
                wmin = weight.min(dim=1)[0]
                wmax = weight.max(dim=1)[0]
                scale_t = ((wmax - wmin) / qmax).clamp(min=1e-8)
                # ``zp = round(-min/scale)`` may be negative (all-positive rows)
                # or exceed qmax (all-negative rows); only ``q`` is clamped to
                # the 8-bit grid. The offset is folded into the int8 storage.
                zp = torch.round(-wmin / scale_t)
                quant_linear.weight_scale.copy_(scale_t)
                quant_linear.weight_zero_point.copy_(zp - 128.0)
                q = ((weight / scale_t.view(-1, 1)).round() + zp.view(-1, 1)).clamp(0, qmax)
            else:
                wmin = weight.min()
                wmax = weight.max()
                scale_v = max(((wmax - wmin) / qmax).item(), 1e-8)
                zp = 0 if scale_v == 1e-8 else round(-wmin.item() / scale_v)
                quant_linear.weight_scale.fill_(scale_v)
                quant_linear.weight_zero_point.fill_(float(zp - 128))
                q = ((weight / scale_v).round() + zp).clamp(0, qmax)
            weight_quantized = (q - 128.0).to(torch.int8)
        elif config.per_channel:
            # Per-channel quantization: a per-row scale vector.
            if scale is None:
                abs_max = weight.abs().max(dim=1)[0]
                qmax = 2 ** (config.bits - 1) - 1
                scale = abs_max / qmax
                scale = scale.clamp(min=1e-8)
            elif isinstance(scale, (int, float)):
                # A caller-supplied scalar scale has no per-channel meaning —
                # as a 0-dim tensor it broadcasts to every row, silently making
                # the dequantization PER-TENSOR while the layer still declares
                # per-channel (every row shares one scale, inflating the
                # quantization-error profile for no structural benefit). Fail
                # fast instead of silently mis-scaling (RIL ISS-135).
                raise ValueError(
                    "per_channel=True requires a per-row scale tensor (or None to compute "
                    f"it), got a scalar {scale!r}. Pass a scale with one value per output "
                    "channel, or set config.per_channel=False."
                )

            # scale is a per-row Tensor (computed or caller-supplied).
            scale_tensor = torch.as_tensor(scale, device=weight.device, dtype=weight.dtype)
            if scale_tensor.numel() != weight.shape[0]:
                raise ValueError(
                    f"per_channel scale must have one value per output channel "
                    f"({weight.shape[0]}), got {scale_tensor.numel()}"
                )
            weight_quantized = (weight / scale_tensor.view(-1, 1)).round().clamp(-128, 127).to(torch.int8)
            quant_linear.weight_scale.copy_(scale_tensor)
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
    scale: float | torch.Tensor | None = None,
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

    Recognizes all quantized layer flavors in the library:
    :class:`QuantizedLinear` (simple PTQ), and the GPTQ / AWQ / SmoothQuant /
    FP8 layers (:class:`~llm.quantization._gptq_layer.GPTQQuantizedLinear`,
    :class:`~llm.quantization._awq_layer.AWQQuantizedLinear`,
    :class:`~llm.quantization._smooth_layer.SmoothQuantLinear`,
    :class:`~llm.quantization._fp8_layer.Fp8QuantizedLinear`). Those replace
    ``nn.Linear`` entirely, so without explicit handling a
    GPTQ/AWQ/Smooth/FP8-quantized model reported zero parameters and zero bytes.

    ``total_params`` counts **true weights**: for 4-bit GPTQ/AWQ layers each
    packed int8 byte stores two int4 weights, so ``total_params`` is the
    unpacked weight count while ``total_bytes`` is the actual (packed) on-disk
    size. Use ``total_params`` for parameter counts and ``total_bytes`` /
    ``size_mb`` for footprint.

    Returns:
        Dictionary with size information.
    """
    # Lazy import to keep this module import-light and avoid a circular
    # dependency (the layer modules import from llm.quantization too).
    from llm.quantization._awq_layer import AWQQuantizedLinear
    from llm.quantization._fp8_layer import Fp8QuantizedLinear
    from llm.quantization._gptq_layer import GPTQQuantizedLinear
    from llm.quantization._smooth_layer import SmoothQuantLinear

    total_params = 0
    total_bytes = 0
    quantized_layers = 0

    for module in model.modules():
        if isinstance(module, (GPTQQuantizedLinear, AWQQuantizedLinear)):
            quantized_layers += 1
            # Packed int8 storage: ``weight_packed`` holds two int4 values
            # per byte for bits=4, or one int8 value per byte for bits=8.
            # ``numel()`` counts *bytes*; the true parameter count is
            # bytes * weights-per-byte (otherwise bits=4 reports half the
            # real weight count — ISS-94). ``total_bytes`` stays the actual
            # packed storage either way.
            packed_attr = cast(torch.Tensor, module.weight_packed)
            weights_per_byte = 2 if getattr(module, "bits", 4) == 4 else 1
            scales_attr = cast(torch.Tensor, module.scales)
            total_params += packed_attr.numel() * weights_per_byte
            total_bytes += packed_attr.numel() * packed_attr.element_size()
            total_bytes += scales_attr.numel() * scales_attr.element_size()
            input_scales = cast(torch.Tensor | None, getattr(module, "input_scales", None))
            if input_scales is not None:
                total_bytes += input_scales.numel() * input_scales.element_size()
            zeros_attr = cast(torch.Tensor | None, getattr(module, "zeros", None))
            if zeros_attr is not None:
                total_bytes += zeros_attr.numel() * zeros_attr.element_size()
            if module.bias is not None:
                total_bytes += module.bias.numel() * module.bias.element_size()
        elif isinstance(module, Fp8QuantizedLinear):
            quantized_layers += 1
            # FP8 weights are real float8 storage (1 byte/weight). The
            # ``weight_scale`` is fp32 per-tensor (1 value) or per-channel.
            weight_attr = module.weight_fp8
            total_params += weight_attr.numel()
            total_bytes += weight_attr.numel() * weight_attr.element_size()  # 1 byte
            scales_attr = module.weight_scale
            total_bytes += scales_attr.numel() * scales_attr.element_size()
            act_attr = cast(torch.Tensor | None, getattr(module, "activation_scale", None))
            if act_attr is not None:
                total_bytes += act_attr.numel() * act_attr.element_size()
            if module.bias is not None:
                total_bytes += module.bias.numel() * module.bias.element_size()
        elif isinstance(module, SmoothQuantLinear):
            quantized_layers += 1
            packed_attr = cast(torch.Tensor, module.weight_packed)
            weight_scales_attr = cast(torch.Tensor, module.weight_scales)
            act_scale_attr = cast(torch.Tensor, module.act_scale)
            total_params += packed_attr.numel()
            total_bytes += packed_attr.numel() * packed_attr.element_size()
            total_bytes += weight_scales_attr.numel() * weight_scales_attr.element_size()
            total_bytes += act_scale_attr.numel() * act_scale_attr.element_size()
            if module.input_scales is not None:
                total_bytes += module.input_scales.numel() * module.input_scales.element_size()
            if module.bias is not None:
                total_bytes += module.bias.numel() * module.bias.element_size()
        elif isinstance(module, QuantizedLinear):
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
