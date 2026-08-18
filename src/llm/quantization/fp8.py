"""FP8 (E4M3/E5M2) post-training quantization.

An FP8 weight + activation PTQ path that mirrors the SmoothQuant structure:
calibration batches are forwarded through the model and per-layer activation
statistics captured via forward hooks, then every ``nn.Linear`` is replaced by
an :class:`~llm.quantization._fp8_layer.Fp8QuantizedLinear` that stores REAL
float8 weights (1 byte/weight) and simulates the fp8 matmul in fp32.

Activation scaling is per-tensor: STATIC (a captured calibration scale per
layer) or DYNAMIC (per-forward absmax). Weights are per-channel (default) or
per-tensor, E4M3FN (default) or E5M2.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from llm.quantization._fp8_layer import FP8_MAX, Fp8QuantizedLinear, quantize_fp8_linear

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fp8Config:
    """Configuration for FP8 quantization.

    Attributes:
        weight_dtype: FP8 format for weights — ``"e4m3"`` (E4M3FN, default,
            3-bit mantissa) or ``"e5m2"`` (E5M2, 2-bit mantissa, wider range).
        per_channel: whether weight scaling is per-output-row (True, default)
            or per-tensor (False).
        activation: ``"static"`` (capture per-layer scales from the
            calibration batches; default) or ``"dynamic"`` (per-forward
            absmax, needs no calibration).

    Note:
        LayerQuantPolicy overrides are NOT wired in v1 — FP8's knobs
        (weight_dtype / per_channel / activation) are not expressible in the
        shared policy fields, so per-layer overrides stay a follow-up rather
        than a silently-ignored option.
    """

    weight_dtype: Literal["e4m3", "e5m2"] = "e4m3"
    per_channel: bool = True
    activation: Literal["static", "dynamic"] = "static"

    def __post_init__(self):
        if self.weight_dtype not in ("e4m3", "e5m2"):
            raise ValueError(f"Fp8Config.weight_dtype must be 'e4m3' or 'e5m2', got {self.weight_dtype!r}.")
        if self.activation not in ("static", "dynamic"):
            raise ValueError(f"Fp8Config.activation must be 'static' or 'dynamic', got {self.activation!r}.")


def _replace_module(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    obj = parent
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], new_module)


def quantize_model_fp8(
    model: nn.Module,
    calib_iter: Iterator[torch.Tensor] | None,
    config: Fp8Config | None = None,
    target_modules: Iterable[str] | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Quantize a model's ``nn.Linear`` layers with FP8 weights + activations.

    Args:
        model: The model to quantize (``nn.Linear`` layers are replaced in
            place by :class:`Fp8QuantizedLinear`).
        calib_iter: Calibration input batches (tensors fed to the model
            forward); required for ``activation="static"``, may be ``None``
            for ``activation="dynamic"``.
        config: Fp8Config (defaults: E4M3, per-channel weights, static acts).
        target_modules: Layer names to quantize (default: all nn.Linear).
        device: Device to run calibration on (default: model's device).

    Returns:
        The (mutated) quantized model.

    Raises:
        ValueError: for empty/invalid calibration, unmatched targets, or a
            layer already quantized.
    """
    config = config or Fp8Config()
    if device is not None:
        model = model.to(device)

    for n, m in model.named_modules():
        if isinstance(m, Fp8QuantizedLinear):
            raise ValueError(f"Layer {n} is already FP8-quantized. Pass a fresh model or unquantize first.")

    linear_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        raise ValueError("model has no nn.Linear modules; nothing to quantize.")

    if target_modules is not None:
        target_set = set(target_modules)
        matched = target_set & {n for n, _ in linear_layers}
        if not matched:
            raise ValueError(f"target_modules {list(target_set)} matched no nn.Linear.")
        targets = [(n, m) for n, m in linear_layers if n in target_set]
    else:
        targets = linear_layers

    # --- static activation scaling: capture per-layer abs-max --------------
    act_scale: dict[str, torch.Tensor] = {}
    if config.activation == "static":
        if calib_iter is None:
            raise ValueError("activation='static' requires calib_iter (calibration batches).")
        batches = list(calib_iter)
        if not batches:
            raise ValueError("calib_iter is empty; need at least 1 batch for activation statistics.")
        captured: dict[str, torch.Tensor] = {n: torch.zeros(1) for n, _ in targets}
        hooks = []

        def make_hook(name: str):
            def hook(_m, inputs, _output):
                x = inputs[0].detach().float()
                amax = x.abs().max()
                if captured[name].numel() == 1:
                    captured[name] = amax
                else:
                    captured[name] = torch.maximum(captured[name], amax)

            return hook

        for n, m in targets:
            hooks.append(m.register_forward_hook(make_hook(n)))

        model.eval()
        with torch.no_grad():
            param_device = next(model.parameters()).device
            for batch in batches:
                try:
                    _ = model(batch.to(param_device))
                except RuntimeError, ValueError, TypeError:
                    logger.warning(
                        "model forward failed during FP8 calibration; layer stats may be partial",
                        exc_info=True,
                    )
        for h in hooks:
            h.remove()

        fmax = FP8_MAX[config.weight_dtype]
        for n, amax in captured.items():
            act_scale[n] = (amax.clamp(min=1e-8) / fmax).to(torch.float32)

    # --- replace linears ----------------------------------------------------
    replaced = 0
    for name, layer in targets:
        if config.activation == "static" and name not in act_scale:
            raise RuntimeError(
                f"static activation calibration produced no scale for target layer {name} "
                "(its forward hook never fired) — use activation='dynamic' or fix the model forward."
            )
        as_scale = act_scale.get(name) if config.activation == "static" else None
        new_layer = quantize_fp8_linear(
            layer,
            dtype_name=config.weight_dtype,
            per_channel=config.per_channel,
            activation_scale=as_scale,
        )
        _replace_module(model, name, new_layer)
        replaced += 1

    logger.info(
        f"FP8-quantized {replaced} linear layers (weight={config.weight_dtype}, "
        f"per_channel={config.per_channel}, activation={config.activation})"
    )
    return model
