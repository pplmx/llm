"""Weight pruning of pretrained models (ROADMAP 13.4 "模型剪枝", RIL TASK-224).

Pruning sparsifies a model's weight matrices by zeroing a fraction of each
``nn.Linear``'s entries, keeping the rest intact. It is complementary to the
QAT / KD compression slices already shipped and runs on any device: the layer
stays architecture-identical (same in/out, same dtype) and the pruned output
is ``x @ (W * M) + b`` where ``M`` is a persistent binary ``weight_mask``.

Two supported policies:

- ``magnitude`` (default): keep the entries with the largest ``|W|`` — the
  classic intuition that large-magnitude weights carry more signal.
- ``random``: keep a uniformly random complement at the target ratio.

``llm-quantize``'s sibling CLI ``llm-prune`` wraps this pass; the resulting
model is saved as a bare ``torch.save`` blob and served through the same
securely-allowlisted path as quantized models (``register_framework_safe_globals``
covers ``llm.quantization``, so :class:`PrunedLinear` loads under
``weights_only=True``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional


def _validate_ratio(ratio: float) -> float:
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"prune ratio must be in (0, 1), got {ratio}")
    return float(ratio)


@dataclass
class PruningConfig:
    """Knobs for a weight-pruning pass.

    Args:
        ratio: fraction of each ``nn.Linear``'s entries to zero (0 < ratio < 1).
        method: ``"magnitude"`` (keep largest |W|, default) or ``"random"``.
        target_modules: optional list of (substring) module names to prune; by
            default every ``nn.Linear`` in the model is pruned.
        random_seed: optional seed so ``"random"`` pruning is reproducible.
    """

    ratio: float = 0.5
    method: str = "magnitude"
    target_modules: list[str] | None = None
    random_seed: int | None = None

    def __post_init__(self) -> None:
        self.ratio = _validate_ratio(self.ratio)
        if self.method not in ("magnitude", "random"):
            raise ValueError(f"unknown prune method {self.method!r}; expected 'magnitude' or 'random'")


class PrunedLinear(nn.Module):
    """An ``nn.Linear`` paired with a persistent binary ``weight_mask``.

    Keeps the original weights as a trainable ``weight`` parameter and a
    read-only ``weight_mask`` buffer; the forward computes ``x @ (W * M) + b``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        weight_mask: torch.Tensor,
        method: str = "magnitude",
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.method = method
        self.weight: torch.Tensor
        self.bias: torch.Tensor | None
        self.weight_mask: torch.Tensor
        self.weight = nn.Parameter(weight.detach().clone())
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None
        self.register_buffer("weight_mask", weight_mask.detach().clone())

    @classmethod
    def from_linear(cls, linear: nn.Linear, weight_mask: torch.Tensor, method: str) -> PrunedLinear:
        """Build a :class:`PrunedLinear` that matches ``linear``'s weight/bias."""
        return cls(
            linear.in_features,
            linear.out_features,
            weight=linear.weight,
            bias=linear.bias,
            weight_mask=weight_mask,
            method=method,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional.linear(x, self.weight * self.weight_mask, self.bias)

    def sparsity(self) -> float:
        """Fraction of entries masked to zero in ``[0, 1]``."""
        flat = self.weight_mask.reshape(-1)
        return 1.0 - (flat.count_nonzero().item() / flat.numel())


def _magnitude_mask(weight: torch.Tensor, ratio: float) -> torch.Tensor:
    """Keep exactly the largest-|W| ``(1 - ratio)`` fraction; 1.0 = keep.

    ``torch.topk`` picks entries BY INDEX, so boundary ties (many entries
    sharing the keep-threshold magnitude — common in sparse/already-pruned
    weights or dead channels, where a large zero-mass sits exactly on the
    boundary) resolve deterministically instead of all surviving. A
    ``abs >= threshold`` style keeps EVERY boundary tie, so a weight whose
    zero-mass exceeds the drop quota prunes NOTHING (achieved sparsity drifts
    silently toward 0) — RIL TASK-309.
    """
    flat = weight.detach().abs().reshape(-1)
    keep = round((1.0 - ratio) * flat.numel())
    keep = max(1, min(keep, flat.numel()))
    _, indices = torch.topk(flat, keep)
    mask = torch.zeros_like(flat)
    mask[indices] = 1.0
    return mask.view_as(weight).to(weight.dtype)


def _random_mask(weight: torch.Tensor, ratio: float, seed: int | None) -> torch.Tensor:
    """Keep a uniformly random (1 - ratio) fraction."""
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    keep = torch.rand(weight.numel(), generator=generator) < (1.0 - ratio)
    return keep.reshape_as(weight).to(weight.dtype)


def _compute_mask(weight: torch.Tensor, config: PruningConfig) -> torch.Tensor:
    if config.method == "magnitude":
        return _magnitude_mask(weight, config.ratio)
    return _random_mask(weight, config.ratio, config.random_seed)


def _replace_module(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a child module by dotted name (getattr/setattr traversal)."""
    parts = name.split(".")
    obj = parent
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], new_module)


def supports_pruning(model: nn.Module) -> bool:
    """Whether the model has any ``nn.Linear`` (prunable) layer."""
    return any(isinstance(m, nn.Linear) and not isinstance(m, PrunedLinear) for m in model.modules())


def prune_model(model: nn.Module, config: PruningConfig) -> float:
    """Replace ``nn.Linear`` with :class:`PrunedLinear` per ``config``.

    Mutates ``model`` in place (the GPTQ/_replace_module pattern). Returns the
    achieved overall sparsity as a fraction in ``[0, 1]``.
    """
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not isinstance(module, PrunedLinear)
    ]
    if not candidates:
        raise ValueError("prune_model found no nn.Linear layers to prune.")
    pruned_any = False
    for name, module in candidates:
        if config.target_modules is not None and not any(tok in name for tok in config.target_modules):
            continue
        mask = _compute_mask(module.weight.detach(), config)
        _replace_module(model, name, PrunedLinear.from_linear(module, mask, config.method))
        pruned_any = True

    if not pruned_any:
        # Matches decompose_model's fail-loud contract: a target_modules filter
        # that matches nothing is a user mistake, not a silent no-op (deep-dive
        # TASK-228).
        raise ValueError("prune_model: target_modules matched no nn.Linear layers to prune.")
    return compute_sparsity(model)


def compute_sparsity(model: nn.Module) -> float:
    """Overall fraction of pruned (zeroed) weight entries across all Linear layers."""
    total = 0
    zeros = 0
    for module in model.modules():
        if isinstance(module, PrunedLinear):
            w = module.weight
            m = module.weight_mask
            total += w.numel()
            zeros += w.numel() - m.count_nonzero().item()
    if total == 0:
        return 0.0
    return zeros / total
