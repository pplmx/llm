"""SmoothQuant (Xiao et al., ICML 2023) post-training quantization.

SmoothQuant makes INT8 weight+activation quantization tractable for LLMs by
migrating the quantization difficulty from activations to weights: per-input-
channel smoothing factors ``s_j = act_max[j]**alpha / w_max[j]**(1-alpha)``
are folded into the weights (``W·s``), and the input is divided by ``s``
before its per-tensor INT8 fake quantization.

This module mirrors the GPTQ / AWQ paths: a frozen config, a stateful
per-layer quantizer, and model-level entry points that capture per-layer
calibration activations through forward hooks.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn

from llm.quantization._policy import LayerQuantPolicy, resolve_layer_policies
from llm.quantization._smooth_layer import SmoothQuantLinear
from llm.quantization.calibration import CalibrationDataCollector, _single_thread_reductions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SmoothQuantConfig:
    """Configuration for SmoothQuant quantization.

    Attributes:
        alpha: Smoothing strength in [0, 1]. alpha=0 pushes all quantization
            difficulty onto activations (weights normalized), alpha=1 pushes
            it onto weights (activations normalized); the paper's default is
            0.5 (balanced).
        search_alpha: If True, grid-search alpha per layer over
            {0.25, 0.5, 0.75, 1.0} using the calibration activations,
            picking the value with the lowest output reconstruction error.
            Requires retaining the calibration batches (more memory).
        bits: Weight bit width. SmoothQuant is an INT8 method in v1 —
            only 8 is accepted; sub-8-bit weight variants are a follow-up.
        group_size: Always -1 (per-channel) in v1 — SmoothQuant weights are
            quantized per output row by design. Present only so
            ``LayerQuantPolicy`` overrides can be validated uniformly.
        sym: If True, symmetric quantization (no zero-point). Asymmetric
            SmoothQuant is not yet implemented.
        act_order: Accepted for ``LayerQuantPolicy`` compatibility but
            ignored — SmoothQuant has no column-reordering step.
        layer_policies: Atomic per-layer override policies. For v1 the only
            meaningful overrides are ``bits=8`` and ``group_size=-1``
            (SmoothQuant weights are per-channel by design); anything else
            fails loudly at quantize time.
    """

    alpha: float = 0.5
    search_alpha: bool = False
    bits: int = 8
    group_size: int = -1
    sym: bool = True
    act_order: bool = False

    # Per-layer atomic override policies (additive; empty tuple = no override).
    layer_policies: tuple[LayerQuantPolicy, ...] = ()

    def __post_init__(self):
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"SmoothQuantConfig.alpha must be in [0, 1], got {self.alpha}.")
        if self.bits != 8:
            raise ValueError(
                f"SmoothQuantConfig.bits must be 8 in v1 (SmoothQuant is an INT8 "
                f"weight+activation method), got {self.bits}. Use AWQ or GPTQ for "
                "sub-8-bit weight-only quantization."
            )
        if self.group_size != -1:
            raise ValueError(f"SmoothQuantConfig.group_size must be -1 (per-channel) in v1, got {self.group_size}.")
        for i, p in enumerate(self.layer_policies):
            if not isinstance(p, LayerQuantPolicy):
                raise TypeError(
                    f"SmoothQuantConfig.layer_policies[{i}] must be LayerQuantPolicy; got {type(p).__name__}."
                )


ALPHA_SEARCH_GRID = (0.25, 0.5, 0.75, 1.0)


def _smoothing_scales(
    act_max: torch.Tensor,
    w_max: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Per-input-channel smoothing factors ``s_j = a**alpha / w**(1-alpha)``.

    Channels with zero activation or zero weight magnitude get scale 1
    (nothing to smooth). Scales are clamped to a small positive floor.

    Args:
        act_max: Per-channel max abs activation [in_features].
        w_max: Per-input-channel max abs weight [in_features].
        alpha: Smoothing strength in [0, 1].

    Returns:
        Per-input-channel smoothing scales [in_features].
    """
    a = act_max.clamp(min=1e-8)
    w = w_max.clamp(min=1e-8)
    s = a**alpha / w ** (1.0 - alpha)
    # Dead channels: no smoothing (scale 1).
    dead = (act_max == 0) | (w_max == 0)
    s = torch.where(dead, torch.ones_like(s), s)
    return s.clamp(min=1e-8)


def _activation_scale(act_max: torch.Tensor, s: torch.Tensor) -> float:
    """Per-tensor INT8 activation scale for the smoothed activations ``x/s``.

    max over samples and channels of ``|x_j| / s_j`` equals
    ``max_j (act_max[j] / s_j)`` because ``s`` is constant per channel —
    so the scale is computable from the activation stats alone.
    """
    return (act_max / s).max().item() / 127.0


def _quantize_layer_components(
    w: torch.Tensor,
    act_max: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a weight matrix at a given smoothing alpha.

    Returns:
        (weight_packed, weight_scales, act_scale, input_scales):
        - weight_packed: int8 flattened weights of ``W·s``.
        - weight_scales: per-output-row fp32 scales [out_features].
        - act_scale: per-tensor fp32 activation scale (scalar).
        - input_scales: per-input-channel smoothing factors [in_features].
    """
    w = w.to(torch.float32)
    act_max = act_max.to(torch.float32)
    w_max = w.abs().max(dim=0)[0]
    s = _smoothing_scales(act_max, w_max, alpha)

    w_s = w * s  # smoothed weights [out_features, in_features]
    weight_scales = (w_s.abs().max(dim=1, keepdim=True)[0] / 127.0).clamp(min=1e-8)
    w_int8 = torch.round(w_s / weight_scales).clamp(-128, 127).to(torch.int8)

    act_scale = torch.tensor(_activation_scale(act_max, s), dtype=torch.float32)
    return w_int8.flatten(), weight_scales, act_scale, s


def _eval_layer_error(
    w: torch.Tensor,
    act_max: torch.Tensor,
    batches: list[torch.Tensor],
    alpha: float,
    bias: torch.Tensor | None,
) -> float:
    """Output reconstruction error of SmoothQuant at a given alpha.

    Uses the exact quantizer math (same scales the packed layer stores) and
    the calibration batches, so the alpha search is honest about the final
    layer's behavior.
    """
    weight_packed, weight_scales, act_scale, s = _quantize_layer_components(w, act_max, alpha)
    w_int = weight_packed.reshape(w.shape).to(torch.float32)
    w_fp = w_int * weight_scales.to(torch.float32)

    total_err = 0.0
    total_n = 0
    for x in batches:
        x = x.to(dtype=torch.float32)
        x_s = x / s
        x_q = torch.clamp(torch.round(x_s / act_scale), -128, 127) * act_scale
        y_q = torch.nn.functional.linear(x_q, w_fp, bias)
        y_ref = torch.nn.functional.linear(x, w, bias)
        total_err += (y_q - y_ref).pow(2).sum().item()
        total_n += y_ref.numel()
    return total_err / total_n


class SmoothQuantQuantizer:
    """Stateful per-layer SmoothQuant processor.

    Lifecycle:
        q = SmoothQuantQuantizer(layer, config)
        for batch in calib_iter_for_this_layer:
            q.add_batch(batch)
        components = q.quantize()   # uses config.alpha or searches alpha
    """

    def __init__(self, layer: nn.Linear, config: SmoothQuantConfig):
        self.config = config
        self.layer = layer
        self.device = layer.weight.device

        self.in_features = layer.weight.shape[1]
        self.act_max = torch.zeros(self.in_features, dtype=torch.float32, device=self.device)
        self.n_samples = 0
        self._batches: list[torch.Tensor] = []

    def add_batch(self, x: torch.Tensor) -> None:
        """Accumulate per-channel max abs activation (and optionally batches)."""
        with _single_thread_reductions():
            x = x.to(device=self.device, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            x = x.reshape(-1, x.shape[-1])
            if x.shape[0] == 0:
                return
            self.act_max = torch.maximum(self.act_max, x.abs().max(dim=0)[0])
            self.n_samples += x.shape[0]
            if self.config.search_alpha:
                self._batches.append(x.detach().clone())

    def quantize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize the layer; returns the packed component tuple."""
        if self.n_samples == 0:
            raise RuntimeError(
                "No calibration data accumulated (n_samples=0). Feed at least one calibration batch via add_batch()."
            )
        if not self.config.sym:
            raise NotImplementedError("Asymmetric SmoothQuant not yet implemented. Use sym=True.")

        with _single_thread_reductions():
            w = self.layer.weight.detach().to(device=self.device, dtype=torch.float32)
            bias = self.layer.bias.detach() if self.layer.bias is not None else None

            if self.config.search_alpha:
                if not self._batches:
                    raise RuntimeError("search_alpha=True requires calibration batches to evaluate alpha candidates.")
                best_alpha = min(
                    ALPHA_SEARCH_GRID,
                    key=lambda a: _eval_layer_error(w, self.act_max, self._batches, a, bias),
                )
                logger.info(f"Layer alpha search picked {best_alpha}")
                alpha = best_alpha
            else:
                alpha = self.config.alpha

            return _quantize_layer_components(w, self.act_max, alpha)


def _replace_module(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a child module by dotted name."""
    parts = name.split(".")
    obj = parent
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], new_module)


def _quantize_linear_with_smoothquant(
    layer: nn.Linear,
    calib_batches: list[torch.Tensor],
    config: SmoothQuantConfig,
) -> SmoothQuantLinear:
    """Run SmoothQuant on a single Linear layer using calibration batches."""
    quantizer = SmoothQuantQuantizer(layer, config)
    for batch in calib_batches:
        quantizer.add_batch(batch)

    weight_packed, weight_scales, act_scale, input_scales = quantizer.quantize()
    return SmoothQuantLinear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=(layer.bias is not None),
        weight_packed=weight_packed,
        weight_scales=weight_scales.to(torch.float16),
        act_scale=act_scale.to(torch.float16),
        sym=config.sym,
        input_scales=input_scales.to(torch.float16),
    ).to(layer.weight.device)


def quantize_model_smoothquant(
    model: nn.Module,
    calib_iter: Iterator[torch.Tensor],
    config: SmoothQuantConfig | None = None,
    target_modules: Iterable[str] | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Quantize a model with SmoothQuant (INT8 weights + activations).

    Args:
        model: nn.Module containing nn.Linear layers to quantize.
        calib_iter: Iterator yielding input tensors for the model forward pass.
        config: SmoothQuantConfig (default: alpha=0.5, INT8 symmetric).
        target_modules: Iterable of fully-qualified layer names to quantize.
            If None, all nn.Linear layers are quantized.
        device: Device to run calibration on (default: model's device).

    Returns:
        The model with nn.Linear layers replaced by SmoothQuantLinear.

    Raises:
        ValueError: If model has no nn.Linear, target_modules unmatched, layer
            already quantized, calibration empty, or a layer policy targets
            an unsupported override.
    """
    config = config or SmoothQuantConfig()
    if device is not None:
        model = model.to(device)

    for n, m in model.named_modules():
        if isinstance(m, SmoothQuantLinear):
            raise ValueError(f"Layer {n} is already SmoothQuant-quantized. Pass a fresh model or unquantize first.")

    linear_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        raise ValueError("model has no nn.Linear modules; nothing to quantize.")

    if target_modules is not None:
        target_set = set(target_modules)
        all_names = {n for n, _ in linear_layers}
        matched = target_set & all_names
        if not matched:
            available = sorted(all_names)[:10]
            raise ValueError(
                f"target_modules {list(target_set)} matched no nn.Linear. "
                f"Available: {available}{'...' if len(all_names) > 10 else ''}"
            )
        targets = [(n, m) for n, m in linear_layers if n in target_set]
    else:
        targets = linear_layers

    calib_batches = list(calib_iter)
    if not calib_batches:
        raise ValueError("calib_iter is empty; need at least 1 batch for activation statistics.")

    captured: dict[str, list[torch.Tensor]] = {n: [] for n, _ in targets}
    hooks = []

    def make_hook(name: str):
        def hook(_module, inputs, _output):
            captured[name].append(inputs[0].detach().clone())

        return hook

    for n, m in targets:
        hooks.append(m.register_forward_hook(make_hook(n)))

    model.eval()
    with torch.no_grad():
        param_device = next(model.parameters()).device
        try:
            for batch in calib_batches[:1]:
                _ = model(batch.to(param_device))
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug(f"Model forward failed during calibration: {e}; falling back to direct layer calls.")

    any_captured = any(len(v) > 0 for v in captured.values())
    if not any_captured:
        for h in hooks:
            h.remove()
        for n, _m in targets:
            captured[n] = [batch.detach().clone() for batch in calib_batches]

    for h in hooks:
        h.remove()

    available_layer_names = {n for n, _ in targets}
    effective_configs = resolve_layer_policies(
        config.layer_policies,
        available_layer_names,
        config,
    )

    for name, layer in targets:
        effective_config = effective_configs.get(name, config)
        # v1 policy constraints (bits=8, group_size=-1) are enforced by
        # SmoothQuantConfig.__post_init__, which dataclasses.replace runs
        # when resolve_layer_policies builds the effective config.
        new_layer = _quantize_linear_with_smoothquant(layer, captured[name], effective_config)
        if layer.bias is not None:
            with torch.no_grad():
                new_layer.bias.copy_(layer.bias.data)
        _replace_module(model, name, new_layer)
        logger.info(
            f"Quantized layer {name}: {layer.weight.shape} → INT8 weight+activation "
            f"(alpha={effective_config.alpha}{'+search' if effective_config.search_alpha else ''})"
        )

    return model


def quantize_model_smoothquant_with_collector(
    model: nn.Module,
    collector: CalibrationDataCollector | Iterable[torch.Tensor],
    n_samples: int,
    config: SmoothQuantConfig | None = None,
    target_modules: Iterable[str] | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Quantize a model using an existing calibration batch source.

    Mirrors ``quantize_model_with_collector`` / ``quantize_model_awq_with_collector``:
    materializes up to ``n_samples`` batches, then funnels them into
    ``quantize_model_smoothquant``.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")

    if not isinstance(collector, Iterable):
        raise TypeError(
            "collector must be an iterable of calibration batches; "
            "CalibrationDataCollector stores activation statistics, not batches"
        )
    batches: list[torch.Tensor] = []
    for i, batch in enumerate(collector):
        if not isinstance(batch, torch.Tensor):
            raise TypeError(f"calibration batches must be tensors, got {type(batch).__name__}")
        batches.append(batch)
        if i + 1 >= n_samples:
            break

    return quantize_model_smoothquant(
        model,
        calib_iter=iter(batches),
        config=config,
        target_modules=target_modules,
        device=device,
    )
