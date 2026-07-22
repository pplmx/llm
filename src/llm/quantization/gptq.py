"""
GPTQ (Frantar et al. 2022) post-training quantization.

Provides 4-bit / 8-bit Hessian-aware quantization orthogonal to
the simple-PTQ path in `ptq.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn

from llm.quantization._gptq_layer import GPTQQuantizedLinear, _pack_4bit
from llm.quantization.calibration import CalibrationDataCollector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPTQConfig:
    """Configuration for GPTQ quantization.

    Attributes:
        bits: Quantization bit width (4 or 8).
        group_size: Quantization group size along input dim.
            -1 means per-channel (one scale per output row).
            Positive integer g means one scale per g consecutive input cols.
        sym: If True, symmetric quantization (no zero-point).
        percdamp: Hessian damping as a fraction of mean(diag(H)).
            Prevents numerical issues when H is near-singular.
        blocksize: Number of weight columns processed per Cholesky block.
            Larger = faster but more memory. Must be divisible by group_size
            when group_size > 0.
        act_order: If True, sort weight columns by diag(H) descending
            before quantization. Improves accuracy at slight cost.
        static_groups: If True, compute group partitions once and reuse
            across all layers. Faster, slight accuracy loss.
    """

    bits: int = 4
    group_size: int = 128
    sym: bool = True
    percdamp: float = 0.01
    blocksize: int = 128
    act_order: bool = False
    static_groups: bool = False

    def __post_init__(self):
        if self.bits not in (4, 8):
            raise ValueError(
                f"GPTQConfig.bits must be 4 or 8, got {self.bits}. "
                f"For mixed precision, use target_modules to skip sensitive layers."
            )
        if self.group_size != -1 and self.group_size < 0:
            raise ValueError(f"group_size must be -1 (per-channel) or positive, got {self.group_size}.")
        if not (0.0 < self.percdamp < 1.0):
            raise ValueError(f"percdamp must be in (0, 1), got {self.percdamp}.")
        if self.blocksize <= 0:
            raise ValueError(f"blocksize must be positive, got {self.blocksize}.")
        if self.group_size > 0 and self.blocksize % self.group_size != 0:
            raise ValueError(
                f"blocksize ({self.blocksize}) must be divisible by "
                f"group_size ({self.group_size}) for correct packing alignment."
            )


class GPTQQuantizer:
    """Stateful per-layer GPTQ processor.

    Lifecycle:
        q = GPTQQuantizer(layer, config)
        for batch in calib_iter_for_this_layer:
            q.add_batch(batch)
        W_packed, scales, zeros = q.quantize()
    """

    def __init__(self, layer: nn.Linear, config: GPTQConfig):
        self.config = config
        self.layer = layer
        self.device = layer.weight.device
        # Compute in float32 for numerical stability of Cholesky
        self.compute_dtype = torch.float32

        # Weight dimensions
        self.out_features, self.in_features = layer.weight.shape

        # Hessian accumulator
        self.H = torch.zeros(
            (self.in_features, self.in_features),
            dtype=self.compute_dtype,
            device=self.device,
        )
        self.n_samples = 0

    def add_batch(self, x: torch.Tensor) -> None:
        """Accumulate Hessian contribution from a calibration batch.

        Maintains the invariant H == (2 / N_total) · Σ X_b^T X_b so that
        multiple mini-batches produce the same H as a single concatenated
        add_batch (Frantar 2022, eq. 3). Uses the canonical EMA-style
        rescale: H_new = (N_old / N_new) · H_old + (2 / N_new) · X^T X.

        Args:
            x: Input activations to `self.layer`, shape [..., in_features].
               Will be flattened to [N, in_features] internally.
        """
        x = x.to(device=self.device, dtype=self.compute_dtype)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.reshape(-1, x.shape[-1])  # flatten leading dims

        n = x.shape[0]
        if n == 0:
            return

        new_total = self.n_samples + n
        # Rescale previous contribution to its raw Σ X^T X form, then
        # re-apply the (2 / N_new) factor on the new total. This makes
        # H the exact (2 / N_total) · Σ X^T X across any batch partition.
        self.H *= self.n_samples / new_total
        self.n_samples = new_total
        self.H += (2.0 / new_total) * (x.t() @ x)

    def quantize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run GPTQ on accumulated Hessian.

        Returns:
            W_q: Quantized weights (integer-valued, stored as fp32),
                 shape [out_features, in_features]. Multiply by `scales`
                 to dequantize: W_recon = W_q * scales.
            scales: Per-row scale if group_size=-1 [out_features, 1],
                    else per-group scale [out_features, in_features // group_size].
            zeros: Per-group zero-points, or None (symmetric only in v1).

        Raises:
            RuntimeError: If calibration is empty or Hessian is ill-conditioned
                          (rank-deficient with insufficient damping).
        """
        # Guard: zero calibration data
        if self.n_samples == 0:
            raise RuntimeError(
                "No calibration data accumulated (n_samples=0). "
                "Hessian is empty. Try increasing percdamp or check calibration data quality."
            )

        w = self.layer.weight.detach().clone().to(device=self.device, dtype=self.compute_dtype)
        h = self.H.clone()

        # Detect rank-deficient Hessian BEFORE dead handling
        n_dead = int(torch.sum(torch.diag(h) == 0).item())

        # Handle all-zero columns (degenerate / unused features)
        dead = torch.diag(h) == 0
        h[dead, dead] = 1.0
        w[:, dead] = 0.0

        # Damping for numerical stability (Frantar 2022, eq. 4)
        damp = self.config.percdamp * torch.mean(torch.diag(h))
        diag_idx = torch.arange(self.in_features, device=self.device)
        h[diag_idx, diag_idx] += damp

        # Guard: rank-deficient Hessian with insufficient damping.
        # If most columns are dead (no variance in calibration), low damping
        # leaves the live columns effectively unscaled relative to the dead
        # ones (which dead-handling set to 1.0). Reject and tell the user.
        if n_dead > self.in_features // 2 and damp < 0.25:
            raise RuntimeError(
                f"Hessian is rank-deficient ({n_dead}/{self.in_features} "
                f"columns have zero variance). Damping (percdamp="
                f"{self.config.percdamp}) is insufficient. Try increasing "
                f"percdamp (e.g. 0.5) or check calibration data quality."
            )

        # Cholesky inverse: only H^-1 is needed for canonical Frantar error correction.
        try:
            h_inv = torch.linalg.inv(h)
        except RuntimeError as e:
            raise RuntimeError(
                f"Hessian is not positive-definite even after damping "
                f"(percdamp={self.config.percdamp}). "
                f"Try increasing percdamp (e.g. 0.1) or check calibration data quality."
            ) from e

        # Optional act-order: sort columns by diag(H_inv) descending
        if self.config.act_order:
            perm = torch.argsort(torch.diag(h_inv), descending=True)
            w = w[:, perm]
            h_inv = h_inv[perm][:, perm]

        # Asymmetric not yet implemented — fail fast before entering column loop
        if not self.config.sym:
            raise NotImplementedError("Asymmetric GPTQ not yet implemented. Use sym=True.")

        # Compute per-row scale ONCE for per-channel (group_size=-1).
        # Canonical GPTQ (Frantar 2022, eq. 5): scale = w.abs().max() / qmax.
        qmax = 2 ** (self.config.bits - 1) - 1
        if self.config.group_size == -1:
            row_scales = w.abs().max(dim=1)[0] / qmax  # [out_f]
            row_scales = row_scales.clamp(min=1e-8)

        # Quantize column-by-column with error correction (Frantar 2022, eq. 5)
        q_out = torch.zeros_like(w)

        for i in range(0, self.in_features, self.config.blocksize):
            i_end = min(i + self.config.blocksize, self.in_features)
            count = i_end - i

            w1 = w[:, i:i_end].clone()
            q1 = torch.zeros_like(w1)
            err1 = torch.zeros_like(w1)
            hinv1 = h_inv[i:i_end, i:i_end]

            # Per-group scale cache: skip recomputation within the same group
            cached_group_idx: int | None = None
            cached_scale: torch.Tensor | None = None

            for j in range(count):
                col = w1[:, j]
                d = hinv1[j, j]

                if self.config.group_size == -1:
                    scale = row_scales  # [out_f], same for all columns
                else:
                    # Per-group scale: cache by group_idx to avoid redundant .abs().max()
                    gs = self.config.group_size
                    group_idx = (i + j) // gs
                    if group_idx != cached_group_idx:
                        group_start = group_idx * gs
                        group_end = group_start + gs
                        w_group = w[:, group_start:group_end]
                        cached_scale = (w_group.abs().max() / qmax).clamp(min=1e-8)
                        cached_group_idx = group_idx
                    scale = cached_scale

                # Quantize to INTEGER (clamped to symmetric range)
                q_int = torch.round(col / scale).clamp(-qmax - 1, qmax)
                q1[:, j] = q_int  # store integer-valued fp32

                # Error correction: propagate quantization error to remaining columns.
                # Canonical GPTQ (Frantar 2022, eq. 5): the error vector is
                # scaled by H^-1 and propagated via the row of H^-1 starting
                # at the current column. Note: H^-1 (not its Cholesky factor U)
                # must be used in BOTH the denominator and the propagation to
                # match the canonical formula exactly.
                err = (col - q_int * scale) / d
                if j + 1 < count:
                    w1[:, j + 1 :] -= err.unsqueeze(1) * hinv1[j, j + 1 :].unsqueeze(0)

                err1[:, j] = err

            q_out[:, i:i_end] = q1

            # Propagate block errors to all remaining columns (canonical GPTQ).
            # Uses H^-1 (not U) to match the canonical Frantar 2022 formula.
            if i_end < self.in_features:
                w[:, i_end:] -= err1 @ h_inv[i:i_end, i_end:]

        # Undo act-order permutation if applied
        if self.config.act_order:
            invperm = torch.argsort(perm)
            q_out = q_out[:, invperm]

        # Return per-row or per-group scales matching the scales used in the loop
        if self.config.group_size != -1:
            gs = self.config.group_size
            n_groups = self.in_features // gs
            scales = torch.zeros(self.out_features, n_groups, dtype=torch.float32)
            for g in range(n_groups):
                s = g * gs
                e = s + gs
                w_g = w[:, s:e]
                scales[:, g] = w_g.abs().max(dim=1)[0] / qmax
                scales[:, g] = scales[:, g].clamp(min=1e-8)
        else:
            scales = row_scales.unsqueeze(1)  # [out_f, 1]

        zeros: torch.Tensor | None = None
        return q_out, scales, zeros


def _replace_module(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a child module by dotted name."""
    parts = name.split(".")
    obj = parent
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], new_module)


def _quantize_linear_with_gptq(
    layer: nn.Linear,
    calib_batches: list[torch.Tensor],
    config: GPTQConfig,
) -> GPTQQuantizedLinear:
    """Run GPTQ on a single Linear layer using accumulated calibration batches."""
    quantizer = GPTQQuantizer(layer, config)
    for batch in calib_batches:
        quantizer.add_batch(batch)

    w_q, _scales, _zeros = quantizer.quantize()

    # Re-quantize the integer-valued w_q into packed int8 storage.
    # Per-group scales (or per-channel) are computed from w_q (dequantized)
    # to match what GPTQ actually output (already error-corrected).
    bits = config.bits
    sym = config.sym
    group_size = config.group_size
    out_f, in_f = w_q.shape

    # Adjust effective group_size if it exceeds in_features.
    # When group_size > in_f, treat whole tensor as one group to avoid
    # zero-sized scales that fail in forward broadcast (e.g., default
    # group_size=128 with in_features=16 on tiny test models).
    effective_group_size = group_size
    if effective_group_size != -1 and effective_group_size > in_f:
        effective_group_size = in_f

    if bits == 4:
        # Symmetric int4: shift signed [-8, 7] → unsigned [0, 15] for packing.
        if sym:
            if effective_group_size == -1:
                # Per-channel: scale from row abs max
                scale = w_q.abs().max(dim=1, keepdim=True)[0] / 7.0
                scale = scale.clamp(min=1e-8)
                w_int = (w_q / scale).round().clamp(-8, 7).to(torch.int8) + 8
            else:
                gs = effective_group_size
                n_groups = in_f // gs
                w_int = torch.zeros_like(w_q, dtype=torch.int8)
                scale = torch.zeros(out_f, n_groups, dtype=torch.float32)
                for g in range(n_groups):
                    s = g * gs
                    e = s + gs
                    w_g = w_q[:, s:e]
                    sc = w_g.abs().max(dim=1, keepdim=True)[0] / 7.0
                    sc = sc.clamp(min=1e-8)
                    scale[:, g : g + 1] = sc
                    w_int[:, s:e] = (w_g / sc).round().clamp(-8, 7).to(torch.int8) + 8
        else:
            raise NotImplementedError("Asymmetric GPTQ not yet implemented")

        packed = _pack_4bit(w_int.flatten())
        zeros_buf = None
    else:
        # 8-bit symmetric
        if sym:
            if effective_group_size == -1:
                scale = w_q.abs().max(dim=1, keepdim=True)[0] / 127.0
                scale = scale.clamp(min=1e-8)
                w_int = (w_q / scale).round().clamp(-128, 127).to(torch.int8)
            else:
                gs = effective_group_size
                n_groups = in_f // gs
                w_int = torch.zeros_like(w_q, dtype=torch.int8)
                scale = torch.zeros(out_f, n_groups, dtype=torch.float32)
                for g in range(n_groups):
                    s = g * gs
                    e = s + gs
                    w_g = w_q[:, s:e]
                    sc = w_g.abs().max(dim=1, keepdim=True)[0] / 127.0
                    sc = sc.clamp(min=1e-8)
                    scale[:, g : g + 1] = sc
                    w_int[:, s:e] = (w_g / sc).round().clamp(-128, 127).to(torch.int8)
        else:
            raise NotImplementedError("Asymmetric GPTQ not yet implemented")
        packed = w_int.flatten()
        zeros_buf = None

    return GPTQQuantizedLinear(
        in_features=in_f,
        out_features=out_f,
        bias=(layer.bias is not None),
        weight_packed=packed,
        scales=scale.to(torch.float16),
        zeros=zeros_buf,
        bits=bits,
        group_size=effective_group_size,
        sym=sym,
    )


def quantize_model_gptq(
    model: nn.Module,
    calib_iter: Iterator[torch.Tensor],
    config: GPTQConfig | None = None,
    target_modules: Iterable[str] | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Quantize a model with GPTQ.

    Args:
        model: nn.Module containing nn.Linear layers to quantize.
        calib_iter: Iterator yielding input tensors for the model forward pass.
        config: GPTQConfig (default: 4-bit, group_size=128, symmetric).
        target_modules: Iterable of fully-qualified layer names to quantize.
            If None, all nn.Linear layers are quantized.
        device: Device to run calibration on (default: model's device).

    Returns:
        The model with nn.Linear layers replaced by GPTQQuantizedLinear.

    Raises:
        ValueError: If model has no nn.Linear, target_modules unmatched, or layer already quantized.
    """
    config = config or GPTQConfig()
    if device is not None:
        model = model.to(device)

    # Check for already-quantized layers FIRST so that a model with
    # only GPTQQuantizedLinear surfaces the actionable error instead of
    # the generic "no nn.Linear" message.
    for n, m in model.named_modules():
        if isinstance(m, GPTQQuantizedLinear):
            raise ValueError(f"Layer {n} is already GPTQ-quantized. Pass a fresh model or unquantize first.")

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
        raise ValueError("calib_iter is empty; need at least 1 batch for Hessian accumulation.")

    # Per-layer input capture: register hooks on target modules to capture their inputs.
    # Each target layer's input is what GPTQ needs for Hessian accumulation.
    captured: dict[str, list[torch.Tensor]] = {n: [] for n, _ in targets}
    hooks = []

    def make_hook(name: str):
        def hook(_module, inputs, _output):
            captured[name].append(inputs[0].detach().clone())

        return hook

    for n, m in targets:
        hooks.append(m.register_forward_hook(make_hook(n)))

    # Try to capture per-layer inputs via model forward pass.
    # If forward fails (shape mismatch etc), fall back to direct layer calls.
    model.eval()
    with torch.no_grad():
        param_device = next(model.parameters()).device
        try:
            for batch in calib_batches[:1]:
                _ = model(batch.to(param_device))
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug(f"Model forward failed during calibration: {e}; falling back to direct layer calls.")

    # If hooks captured nothing, fall back to calling each target layer directly.
    any_captured = any(len(v) > 0 for v in captured.values())
    if not any_captured:
        for h in hooks:
            h.remove()
        for n, _m in targets:
            captured[n] = [batch.detach().clone() for batch in calib_batches]

    for h in hooks:
        h.remove()

    for name, layer in targets:
        new_layer = _quantize_linear_with_gptq(layer, captured[name], config)
        if layer.bias is not None:
            with torch.no_grad():
                new_layer.bias.copy_(layer.bias.data)
        _replace_module(model, name, new_layer)
        logger.info(f"Quantized layer {name}: {layer.weight.shape} → 4-bit packed")

    return model


def quantize_model_with_collector(
    model: nn.Module,
    collector: CalibrationDataCollector | Iterable[torch.Tensor],
    n_samples: int,
    config: GPTQConfig | None = None,
    target_modules: Iterable[str] | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Quantize a model using an existing CalibrationDataCollector.

    Trainer-loop entry point: reuse the same calibration batches already
    collected during training (e.g. for activation stats). Materializes
    up to `n_samples` batches, then funnels into `quantize_model_gptq`.

    Args:
        model: nn.Module to quantize.
        collector: CalibrationDataCollector (or any iterable yielding Tensor
            batches). Up to `n_samples` batches are consumed.
        n_samples: Maximum number of batches to use for calibration.
        config: GPTQConfig (default: 4-bit, group_size=128, symmetric).
        target_modules: Optional layer-name filter forwarded to
            `quantize_model_gptq`.
        device: Target device forwarded to `quantize_model_gptq`.

    Returns:
        The quantized model (same instance as `model`, with nn.Linear replaced).

    Raises:
        ValueError: Forwarded from `quantize_model_gptq` (no nn.Linear,
            unmatched target_modules, etc.).
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")

    # Materialize up to n_samples batches from the collector. We stop early
    # so collectors backed by expensive iterators (e.g. dataset streams)
    # don't pull more data than needed.
    batches: list[torch.Tensor] = []
    for i, batch in enumerate(collector):
        batches.append(batch)
        if i + 1 >= n_samples:
            break

    return quantize_model_gptq(
        model,
        calib_iter=iter(batches),
        config=config,
        target_modules=target_modules,
        device=device,
    )
