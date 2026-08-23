"""Low-rank (SVD U-V) decomposition of pretrained linear weights (ROADMAP 13.4).

The final 模型压缩 slice (after QAT / KD / pruning): factorize each pretrained
``nn.Linear`` weight matrix ``W (out x in)`` as ``W ≈ U @ V`` with a reduced
intermediate rank ``r``. Using the truncated SVD
``W = U_s S Vh``, ``U = U_s[:, :r] * sqrt(S[:r])`` and ``V = sqrt(S[:r]) * Vh[:r]``
so ``U @ V`` best-approximates ``W`` in Frobenius norm (Eckart-Young).

As with :mod:`llm.quantization.prune`, the resulting model keeps its
architecture and dtype and runs on any device; the layer's forward is
``y = (x @ V^T) @ U^T + b``. ``llm-decompose`` wraps this pass and the
``LowRankLinear`` class is covered by the ``weights_only`` safe-globals
allowlist (``llm.quantization``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional


@dataclass
class LowRankConfig:
    """Knobs for an SVD low-rank decomposition pass.

    Exactly one of ``rank`` (explicit) or ``rank_ratio`` (fraction of the
    ``min(out, in)`` singular values to keep) must be set.
    """

    rank: int | None = None
    rank_ratio: float | None = None
    target_modules: list[str] | None = None

    def __post_init__(self) -> None:
        uses_rank = self.rank is not None
        uses_ratio = self.rank_ratio is not None
        if uses_rank == uses_ratio:
            raise ValueError("LowRankConfig: set exactly one of rank or rank_ratio")
        if uses_rank and self.rank is not None and self.rank <= 0:
            raise ValueError(f"LowRankConfig: rank must be > 0, got {self.rank}")
        if uses_ratio and not 0.0 < self.rank_ratio <= 1.0:
            raise ValueError(f"LowRankConfig: rank_ratio must be in (0, 1], got {self.rank_ratio}")


class LowRankLinear(nn.Module):
    """An ``nn.Linear``-shaped layer whose weight is stored as ``u (out x r)``
    and ``v (r x in)``, so the effective weight is ``u @ v``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        u: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if v.size(1) != in_features or u.size(0) != out_features or u.size(1) != v.size(0):
            raise ValueError(
                f"low-rank shapes inconsistent: u{u.shape}, v{v.shape}, in={in_features}, out={out_features}"
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(u.size(1))
        self.u: torch.Tensor
        self.v: torch.Tensor
        self.bias: torch.Tensor | None
        self.u = nn.Parameter(u.detach().clone())
        self.v = nn.Parameter(v.detach().clone())
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (x @ V^T) @ U^T + b == x @ (U @ V)^T + b
        return functional.linear(functional.linear(x, self.v), self.u, self.bias)

    def reconstruct(self) -> torch.Tensor:
        """The full-rank weight approximation ``u @ v`` (out x in)."""
        return self.u @ self.v

    def compression_ratio(self) -> float:
        """Original weights / stored low-rank factors (>1 when r < min/2-ish)."""
        out, in_ = self.out_features, self.in_features
        original = out * in_
        stored = out * self.rank + self.rank * in_
        if stored == 0:
            return float("inf")
        return original / stored


def _auto_rank(weight: torch.Tensor, ratio: float) -> int:
    return max(1, round(ratio * min(weight.shape)))


def _svd_factors(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (u, v) such that ``u @ v`` is the best rank-``rank`` approx of ``weight``."""
    u_s, singular, vh = torch.linalg.svd(weight, full_matrices=False)
    s = singular[:rank]
    u = u_s[:, :rank] * s.sqrt().unsqueeze(0)
    v = s.sqrt().unsqueeze(-1) * vh[:rank]
    return u, v


def decompose_layer(linear: nn.Linear, rank: int) -> LowRankLinear:
    """Build a :class:`LowRankLinear` from an ``nn.Linear`` at the given rank."""
    u, v = _svd_factors(linear.weight.detach(), rank)
    return LowRankLinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        u=u,
        v=v,
        bias=linear.bias.detach() if linear.bias is not None else None,
    )


def _replace_module(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    obj = parent
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], new_module)


def _relative_error(approx: torch.Tensor, original: torch.Tensor) -> float:
    denom = torch.linalg.norm(original)
    if denom == 0:
        return 0.0
    return torch.linalg.norm(approx - original).item() / denom.item()


def decompose_model(model: nn.Module, config: LowRankConfig) -> dict:
    """Replace each ``nn.Linear`` with a low-rank :class:`LowRankLinear`.

    Mutates ``model`` in place. Returns a stats dict with ``compression_ratio``,
    ``relative_error`` (mean over decomposed layers) and ``layers`` (name, rank).
    """
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not isinstance(module, LowRankLinear)
    ]
    if not candidates:
        raise ValueError("decompose_model found no nn.Linear layers to decompose.")

    layers: list[tuple[str, int]] = []
    errors: list[float] = []
    for name, module in candidates:
        if config.target_modules is not None and not any(tok in name for tok in config.target_modules):
            continue
        if config.rank is not None:
            rank = config.rank
        else:
            rank_ratio = config.rank_ratio
            if rank_ratio is None:  # pragma: no cover - guaranteed by __post_init__
                raise ValueError("LowRankConfig requires rank or rank_ratio")
            rank = _auto_rank(module.weight.detach(), rank_ratio)
        rank = min(rank, module.weight.size(0), module.weight.size(1))
        new_layer = decompose_layer(module, rank)
        errors.append(_relative_error(new_layer.reconstruct(), module.weight.detach()))
        layers.append((name, rank))
        _replace_module(model, name, new_layer)

    if not layers:
        raise ValueError("decompose_model: target_modules matched no nn.Linear layers.")

    return {
        "compression_ratio": compute_compression(model),
        "relative_error": (sum(errors) / len(errors)) if errors else 0.0,
        "layers": layers,
    }


def compute_compression(model: nn.Module) -> float:
    """Overall (original weights) / (stored low-rank factors) across layers."""
    orig = 0
    stored = 0
    for module in model.modules():
        if isinstance(module, LowRankLinear):
            orig += module.out_features * module.in_features
            stored += module.out_features * module.rank + module.rank * module.in_features
    if stored == 0:
        return 0.0  # no low-rank layers
    return orig / stored
