"""GRPO (Group Relative Policy Optimization) loss + group advantages (TASK-229).

GRPO (Shao et al. 2024) aligns a policy by giving each response a **group-relative**
advantage — the z-score of its reward within the group of ``G`` responses sampled
for the same prompt — and optimizing a clipped importance-weighted policy-ratio
objective with an optional KL-to-reference penalty. This module provides the two
pure pieces (no rollout/sampling infra): :func:`group_advantages` and
:class:`GRPOLoss`, which the ``--task grpo`` task drives with reward-scored
synthetic groups on CPU.

Unclipped objective per token: ``min(ratio, clip(ratio)) * A`` (surrogate), with
``ratio = exp(log_p_policy - log_p_old)`` and ``A`` the (detached) group advantage.
Penalty (K3 estimator): ``exp(ref - pol) - (ref - pol) - 1``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Z-score each reward within its row-group.

    ``rewards`` is ``[num_groups, G]`` (one row per prompt, ``G`` responses).
    Returns the same shape, where each element is ``(r - mean_row) / (std_row + eps)``.
    """
    if rewards.ndim != 2:
        raise ValueError(f"group_advantages expects [num_groups, G] rewards, got shape {tuple(rewards.shape)}")
    rewards = rewards.float()
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, unbiased=False, keepdim=True)
    return (rewards - mean) / (std + eps)


class GRPOLoss(nn.Module):
    """Clipped importance-weighted GRPO policy loss with optional KL-to-reference.

    Args:
        clip_eps: PPO-style ratio clipping range (1-eps, 1+eps).
        kl_beta: weight of the K3 KL-to-reference penalty (0 disables it).
    """

    def __init__(self, clip_eps: float = 0.2, kl_beta: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= clip_eps < 1.0:
            raise ValueError(f"clip_eps must be in [0, 1), got {clip_eps}")
        if kl_beta < 0:
            raise ValueError(f"kl_beta must be >= 0, got {kl_beta}")
        self.clip_eps = float(clip_eps)
        self.kl_beta = float(kl_beta)

    def _kl(self, ref_log_probs: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        diff = ref_log_probs - log_probs
        return torch.mean(torch.exp(diff) - diff - 1.0)

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Return the GRPO loss.

        ``log_probs`` / ``old_log_probs`` / ``ref_log_probs`` are ``[B, T]``
        per-token log-probs; ``advantages`` is ``[B]`` (per-*response*, broadcast
        over its tokens) and must be detached.
        """
        ratio = torch.exp(log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        adv = advantages.detach().unsqueeze(-1)  # [B, 1]
        surrogate = (torch.min(ratio, clipped) * adv).mean()
        penalty = self.kl_beta * self._kl(ref_log_probs, log_probs)
        return -surrogate + penalty
