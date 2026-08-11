"""AWQ (Lin et al., MLSys 2024) post-training quantization.

Activation-aware Weight Quantization protects the ~1% of weight channels
that dominate quantization error: per-input-channel scales ``s`` are
searched over a power-of-two grid to minimize the activation-weighted
reconstruction error of the layer output, then the layer is group-quantized
from ``W·s`` with the scale compensation ``x/s`` applied at forward time.

This module mirrors the GPTQ path in ``gptq.py``: a frozen config, a
stateful per-layer quantizer, and model-level entry points that capture
per-layer calibration activations through forward hooks.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn

from llm.quantization._awq_layer import AWQQuantizedLinear
from llm.quantization._gptq_layer import _pack_4bit
from llm.quantization._policy import LayerQuantPolicy, resolve_layer_policies
from llm.quantization.calibration import CalibrationDataCollector, _single_thread_reductions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AWQConfig:
    """Configuration for AWQ quantization.

    Attributes:
        bits: Quantization bit width (4 or 8).
        group_size: Quantization group size along input dim.
            -1 means per-channel (one scale per output row).
        sym: If True, symmetric quantization (no zero-point). Asymmetric
            AWQ (zero points) is not yet implemented.
        n_grid: Number of power-of-two scale candidates in the grid search,
            centered on 1 (ratios ``2**(-n_grid//2) ... 2**(n_grid//2)``).
            Larger grids find better scales at more search cost.
        clip_ratio: Optional weight clipping ratio in (0, 0.5]. When set,
            each layer's weights are clipped to
            ``[min + rho*(max-min), max - rho*(max-min)]`` before the scale
            search, suppressing outlier magnitudes. None = no clipping.
        layer_policies: Atomic per-layer override policies (algorithm-agnostic
            ``LayerQuantPolicy`` tuples), same semantics as GPTQ (see ADR-008).
    """

    bits: int = 4
    group_size: int = 128
    sym: bool = True
    n_grid: int = 20
    clip_ratio: float | None = None

    # Per-layer atomic override policies (additive; empty tuple = no override).
    layer_policies: tuple[LayerQuantPolicy, ...] = ()

    def __post_init__(self):
        if self.bits not in (4, 8):
            raise ValueError(
                f"AWQConfig.bits must be 4 or 8, got {self.bits}. "
                "For mixed precision, use target_modules to skip sensitive layers."
            )
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError(f"group_size must be -1 (per-channel) or positive, got {self.group_size}.")
        if self.n_grid < 1:
            raise ValueError(f"n_grid must be >= 1, got {self.n_grid}.")
        if self.clip_ratio is not None and not (0.0 < self.clip_ratio <= 0.5):
            raise ValueError(f"clip_ratio must be in (0, 0.5] or None, got {self.clip_ratio}.")
        for i, p in enumerate(self.layer_policies):
            if not isinstance(p, LayerQuantPolicy):
                raise TypeError(f"AWQConfig.layer_policies[{i}] must be LayerQuantPolicy; got {type(p).__name__}.")


def _group_quantize_dequant(
    w: torch.Tensor,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    """Symmetric group quantization round-trip (dequantized output).

    Must match the packed-storage math in ``_quantize_linear_with_awq`` so
    the scale search evaluates the exact error the final layer will have:
    per-group scale = max|w| / qmax, integers = round(w/scale) clamped to the
    symmetric range.

    Args:
        w: Weight tensor [out_features, in_features] (fp32).
        bits: 4 or 8.
        group_size: -1 (per-channel) or positive int.

    Returns:
        Dequantized weights with the same shape as ``w``.
    """
    qmax = 2 ** (bits - 1) - 1
    _out_f, in_f = w.shape

    effective_group_size = in_f if group_size == -1 or group_size > in_f else group_size
    n_groups = in_f // effective_group_size

    dequant = torch.zeros_like(w)
    for g in range(n_groups):
        s = g * effective_group_size
        e = s + effective_group_size
        w_g = w[:, s:e]
        scale = (w_g.abs().max(dim=1, keepdim=True)[0] / qmax).clamp(min=1e-8)
        dequant[:, s:e] = torch.round(w_g / scale).clamp(-qmax - 1, qmax) * scale
    return dequant


def _search_scale(
    w: torch.Tensor,
    act_mean: torch.Tensor,
    bits: int,
    group_size: int,
    n_grid: int,
) -> torch.Tensor:
    """Grid-search per-input-channel AWQ scales minimizing activation-weighted error.

    Candidate scales are powers of two centered on 1 (``2**k`` for
    ``k in -n_grid//2 .. n_grid//2``), AWQ paper eq. 5. The search exploits
    the structural fact that group quantization decouples along the input
    dimension: the per-group (per output row) scale depends only on the
    columns of that group, so each input channel can be optimized greedily
    within its group. Each candidate is evaluated with the **exact** group
    quantizer used at packing time, and the group reconstruction error is
    weighted by each channel's mean absolute activation — channels with
    larger activations are "salient" and dominate the objective, so the
    search trades a small quantization penalty on ordinary channels for a
    large error reduction on salient ones.

    Args:
        w: Weight tensor [out_features, in_features] (fp32).
        act_mean: Mean absolute activation per input channel [in_features].
        bits: 4 or 8.
        group_size: -1 (per-channel) or positive int.
        n_grid: Number of grid candidates.

    Returns:
        Per-input-channel scale ``s`` [in_features] (clamped >= 1e-8).
    """
    w = w.to(torch.float32)
    _out_f, in_f = w.shape
    qmax = 2 ** (bits - 1) - 1

    half = n_grid // 2
    grid = 2.0 ** torch.arange(-half, n_grid - half, dtype=torch.float32, device=w.device)

    # Column groups (disjoint in the input dim); group_size=-1 → one group.
    if group_size == -1:
        groups: list[torch.Tensor] = [torch.arange(in_f, device=w.device)]
    else:
        gs = min(group_size, in_f)
        groups = [torch.arange(s, min(s + gs, in_f), device=w.device) for s in range(0, in_f, gs)]

    scale = torch.ones(in_f, dtype=torch.float32, device=w.device)
    with _single_thread_reductions():
        # Greedy coordinate refinement: 3 passes let a raised group max (from
        # a scaled-up salient channel) be re-evaluated against the others.
        for _pass in range(3):
            for cols in groups:
                w_g = w[:, cols]  # [out_f, gs]
                act_g = act_mean[cols]  # [gs]
                s_g = scale[cols]  # [gs]
                for j in range(len(cols)):
                    best_s = s_g[j].item()
                    best_err = float("inf")
                    for cand in grid:
                        s_try = s_g.clone()
                        s_try[j] = cand
                        scaled = w_g * s_try  # [out_f, gs]
                        # Per-output-row group scale, exactly as at packing time.
                        delta = (scaled.abs().max(dim=1, keepdim=True)[0] / qmax).clamp(min=1e-8)
                        recon = torch.round(scaled / delta).clamp(-qmax - 1, qmax) * delta / s_try
                        # Activation-weighted group reconstruction error.
                        err = (((w_g - recon) ** 2) * act_g).sum().item()
                        if err < best_err:
                            best_err = err
                            best_s = cand
                    s_g[j] = best_s
                    scale[cols[j]] = best_s

    return scale.clamp(min=1e-8)


class AWQQuantizer:
    """Stateful per-layer AWQ processor.

    Lifecycle:
        q = AWQQuantizer(layer, config)
        for batch in calib_iter_for_this_layer:
            q.add_batch(batch)
        scale = q.search_scale()   # per-input-channel AWQ scale
        packed, scales, effective_group_size = q.quantize(scale)
    """

    def __init__(self, layer: nn.Linear, config: AWQConfig):
        self.config = config
        self.layer = layer
        self.device = layer.weight.device
        self.compute_dtype = torch.float32

        self.out_features, self.in_features = layer.weight.shape

        # Group quantization requires the *effective* group size (clamped to
        # in_features, so a group larger than the row is a single group) to
        # divide in_features (the packing path builds
        # ``in_features // group_size`` groups and slices columns
        # [g*gs:(g+1)*gs)); reject non-divisible effective group sizes up
        # front with a clear error instead of a late packing bug.
        gs = min(self.config.group_size, self.in_features)
        if self.config.group_size != -1 and self.in_features % gs != 0:
            raise ValueError(
                f"group_size ({self.config.group_size}) must divide in_features "
                f"({self.in_features}); got remainder {self.in_features % gs}. "
                "Use group_size=-1 (per-channel) or a divisor of in_features."
            )

        self.act_abs_sum = torch.zeros(
            self.in_features,
            dtype=self.compute_dtype,
            device=self.device,
        )
        self.n_samples = 0

    def add_batch(self, x: torch.Tensor) -> None:
        """Accumulate per-input-channel absolute activation sums.

        Args:
            x: Input activations to `self.layer`, shape [..., in_features].
                Leading dims are flattened; only the mean absolute activation
                per channel is retained (that is all the scale search needs).
        """
        x = x.to(device=self.device, dtype=self.compute_dtype)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.reshape(-1, x.shape[-1])  # flatten leading dims

        n = x.shape[0]
        if n == 0:
            return
        self.act_abs_sum += x.abs().sum(dim=0)
        self.n_samples += n

    def act_mean(self) -> torch.Tensor:
        """Mean absolute activation per input channel [in_features]."""
        if self.n_samples == 0:
            raise RuntimeError(
                "No calibration data accumulated (n_samples=0). Feed at least one calibration batch via add_batch()."
            )
        return self.act_abs_sum / self.n_samples

    def search_scale(self) -> torch.Tensor:
        """Run the activation-aware grid search; returns per-channel scale [in_f]."""
        if not self.config.sym:
            raise NotImplementedError("Asymmetric AWQ not yet implemented. Use sym=True.")

        w = self.layer.weight.detach().clone().to(device=self.device, dtype=self.compute_dtype)
        act = self.act_mean()

        if self.config.clip_ratio is not None:
            rho = self.config.clip_ratio
            w_min = w.min()
            w_max = w.max()
            lower = w_min + rho * (w_max - w_min)
            upper = w_max - rho * (w_max - w_min)
            w = w.clamp(lower, upper)

        return _search_scale(
            w,
            act,
            bits=self.config.bits,
            group_size=self.config.group_size,
            n_grid=self.config.n_grid,
        )

    def quantize(self, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Quantize ``W·scale`` into packed storage.

        Returns:
            weight_packed: int8 packed weights.
            scales: per-group (or per-channel) fp32 scales.
            effective_group_size: group size actually used for packing
                (never larger than ``in_features``; -1 for per-channel).
        """
        w = self.layer.weight.detach().clone().to(device=self.device, dtype=self.compute_dtype)
        if self.config.clip_ratio is not None:
            rho = self.config.clip_ratio
            w_min = w.min()
            w_max = w.max()
            lower = w_min + rho * (w_max - w_min)
            upper = w_max - rho * (w_max - w_min)
            w = w.clamp(lower, upper)
        w = w * scale  # [out_f, in_f] * [in_f] broadcasts per input channel
        packed, scales, effective_group_size = _pack_weights(w, self.config.bits, self.config.group_size)
        return packed, scales, effective_group_size


def _pack_weights(
    w: torch.Tensor,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pack scaled weights into int8 storage with per-group scales.

    Symmetric only: 4-bit values are stored unsigned [0, 15] (signed range
    shifted by +8), 8-bit values are stored as signed int8 directly. The
    per-group scale is max|w| / qmax, matching ``_group_quantize_dequant``.

    Returns:
        (weight_packed, scales, effective_group_size) — scales fp32; the
        effective group size (never larger than ``in_features``) must be
        stored in the layer so forward dequantization matches the packing.
    """
    out_f, in_f = w.shape
    effective_group_size = group_size
    if effective_group_size != -1 and effective_group_size > in_f:
        effective_group_size = in_f

    qmax = 2 ** (bits - 1) - 1
    if effective_group_size == -1:
        scale = (w.abs().max(dim=1, keepdim=True)[0] / qmax).clamp(min=1e-8)
        w_int = torch.round(w / scale).clamp(-qmax - 1, qmax)
    else:
        n_groups = in_f // effective_group_size
        w_int = torch.zeros_like(w)
        scale = torch.zeros(out_f, n_groups, dtype=torch.float32, device=w.device)
        for g in range(n_groups):
            s = g * effective_group_size
            e = s + effective_group_size
            w_g = w[:, s:e]
            sc = (w_g.abs().max(dim=1, keepdim=True)[0] / qmax).clamp(min=1e-8)
            scale[:, g : g + 1] = sc
            w_int[:, s:e] = torch.round(w_g / sc).clamp(-qmax - 1, qmax)

    packed = _pack_4bit(w_int.to(torch.int8).flatten() + 8) if bits == 4 else w_int.to(torch.int8).flatten()
    return packed, scale, effective_group_size


def _replace_module(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a child module by dotted name."""
    parts = name.split(".")
    obj = parent
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], new_module)


def _quantize_linear_with_awq(
    layer: nn.Linear,
    calib_batches: list[torch.Tensor],
    config: AWQConfig,
) -> AWQQuantizedLinear:
    """Run AWQ on a single Linear layer using accumulated calibration batches."""
    quantizer = AWQQuantizer(layer, config)
    for batch in calib_batches:
        quantizer.add_batch(batch)

    scale = quantizer.search_scale()
    packed, scales, effective_group_size = quantizer.quantize(scale)

    return AWQQuantizedLinear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=(layer.bias is not None),
        weight_packed=packed,
        scales=scales.to(torch.float16),
        bits=config.bits,
        group_size=effective_group_size,
        sym=config.sym,
        input_scales=scale.to(torch.float16),
    ).to(layer.weight.device)


def quantize_model_awq(
    model: nn.Module,
    calib_iter: Iterator[torch.Tensor],
    config: AWQConfig | None = None,
    target_modules: Iterable[str] | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Quantize a model with AWQ.

    Args:
        model: nn.Module containing nn.Linear layers to quantize.
        calib_iter: Iterator yielding input tensors for the model forward pass.
        config: AWQConfig (default: 4-bit, group_size=128, symmetric, n_grid=20).
        target_modules: Iterable of fully-qualified layer names to quantize.
            If None, all nn.Linear layers are quantized.
        device: Device to run calibration on (default: model's device).

    Returns:
        The model with nn.Linear layers replaced by AWQQuantizedLinear.

    Raises:
        ValueError: If model has no nn.Linear, target_modules unmatched, layer
            already quantized, or calibration is empty.
    """
    config = config or AWQConfig()
    if device is not None:
        model = model.to(device)

    for n, m in model.named_modules():
        if isinstance(m, AWQQuantizedLinear):
            raise ValueError(f"Layer {n} is already AWQ-quantized. Pass a fresh model or unquantize first.")

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

    # Per-layer input capture via forward hooks (same mechanism as GPTQ).
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
            # Feed EVERY calibration batch so each layer's activation stats
            # cover the full calibration set.  Previously only
            # calib_batches[0] was forwarded, silently dropping later batches
            # while the fallback path below used all of them.
            for batch in calib_batches:
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
        new_layer = _quantize_linear_with_awq(layer, captured[name], effective_config)
        if layer.bias is not None:
            with torch.no_grad():
                new_layer.bias.copy_(layer.bias.data)
        _replace_module(model, name, new_layer)
        logger.info(
            f"Quantized layer {name}: {layer.weight.shape} → "
            f"{effective_config.bits}-bit, group_size={effective_config.group_size}"
        )

    return model


def quantize_model_awq_with_collector(
    model: nn.Module,
    collector: CalibrationDataCollector | Iterable[torch.Tensor],
    n_samples: int,
    config: AWQConfig | None = None,
    target_modules: Iterable[str] | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Quantize a model using an existing calibration batch source.

    Trainer-loop entry point mirroring ``quantize_model_with_collector``:
    materializes up to ``n_samples`` batches, then funnels them into
    ``quantize_model_awq``.

    Args:
        model: nn.Module to quantize.
        collector: Iterable yielding Tensor batches. Up to ``n_samples``
            batches are consumed.
        n_samples: Maximum number of batches to use for calibration.
        config: AWQConfig (default: 4-bit, group_size=128, symmetric).
        target_modules: Optional layer-name filter forwarded to
            ``quantize_model_awq``.
        device: Target device forwarded to ``quantize_model_awq``.

    Returns:
        The quantized model (same instance as ``model``).
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

    return quantize_model_awq(
        model,
        calib_iter=iter(batches),
        config=config,
        target_modules=target_modules,
        device=device,
    )
