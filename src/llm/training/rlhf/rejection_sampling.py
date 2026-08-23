"""Rejection sampling for RL alignment (ROADMAP 11.4 / RIL TASK-230).

Rejection sampling (best-of-N / top-K): sample ``N`` responses per prompt, score
each with a reward, then **keep the highest-reward subset** (top-K or
above-threshold) and fine-tune the policy on those selected responses. This
module provides the pure selection + statistics pieces (no generation/serving
infra); the ``RejectionSampleDataModule`` consumes its mask to emit the kept
responses as an SFT-style supervised set.
"""

from __future__ import annotations

import torch


def select_top_k(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Return a boolean mask keeping the ``k`` highest-scored entries (1-D ``scores``)."""
    if scores.ndim != 1:
        raise ValueError(f"select_top_k expects 1-D scores, got shape {tuple(scores.shape)}")
    if scores.numel() == 0:
        raise ValueError("select_top_k expects a non-empty scores tensor")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    k = min(k, scores.numel())
    _, top_idx = torch.topk(scores, k)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask[top_idx] = True
    return mask


def select_above_threshold(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Return a boolean mask keeping entries with ``score >= threshold``."""
    if scores.ndim != 1:
        raise ValueError(f"select_above_threshold expects 1-D scores, got shape {tuple(scores.shape)}")
    return scores >= threshold


def rejection_sample(
    scores: torch.Tensor,
    *,
    k: int | None = None,
    threshold: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Select the high-reward subset and report reward statistics.

    Exactly one of ``k`` (top-K) or ``threshold`` must be given.

    Returns ``(mask, stats)`` where ``mask`` is a bool tensor over the flat scores
    and ``stats`` has ``kept_mean_reward``, ``base_mean_reward``, ``kept_fraction``.
    """
    if (k is None) == (threshold is None):
        raise ValueError("rejection_sample: supply exactly one of k or threshold")
    if scores.ndim != 1:
        raise ValueError(f"rejection_sample expects 1-D scores, got shape {tuple(scores.shape)}")
    if scores.numel() == 0:
        raise ValueError("rejection_sample expects a non-empty scores tensor")

    if k is not None:
        mask = select_top_k(scores, k)
    else:
        threshold_value = threshold
        if threshold_value is None:  # pragma: no cover - guarded by the XOR check above
            raise ValueError("rejection_sample: threshold must be set (k unset)")
        mask = select_above_threshold(scores, threshold_value)
    base_mean = float(scores.float().mean())
    kept = scores.float()[mask]
    kept_mean = float(kept.mean()) if kept.numel() > 0 else 0.0
    return mask, {
        "kept_mean_reward": kept_mean,
        "base_mean_reward": base_mean,
        "kept_fraction": float(mask.float().mean()),
    }
