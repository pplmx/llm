"""AI-feedback preference labeling for DPO (ROADMAP 11.4 / RIL TASK-231).

RLAIF: instead of human labels, an AI/rule-based *judge* decides which of two
responses is preferred; the resulting (chosen, rejected) pairs are consumed by
the existing DPO task. For CPU verifiability the judge here is a deterministic
rule/scalar (`TargetTokenJudge`) — a stand-in for a real judge model.
"""

from __future__ import annotations

import abc

import torch


class PreferenceJudge(abc.ABC):
    """Scores a batch of responses and orders pairs into chosen/rejected."""

    @abc.abstractmethod
    def score_batch(self, responses: torch.Tensor) -> torch.Tensor:
        """Return a ``[B]`` score per response (higher = preferred)."""


class TargetTokenJudge(PreferenceJudge):
    """Rule/scalar judge: a response is preferred iff it ends with a target token.

    Args:
        target_token: the (last-position) token that marks a good response.
    """

    def __init__(self, target_token: int = 0) -> None:
        self.target_token = int(target_token)

    def score_batch(self, responses: torch.Tensor) -> torch.Tensor:
        if responses.ndim != 2:
            raise ValueError(f"TargetTokenJudge expects [B, L] responses, got shape {tuple(responses.shape)}")
        return (responses[:, -1] == self.target_token).float()


def prefer_batch(a: torch.Tensor, b: torch.Tensor, judge: PreferenceJudge) -> tuple[torch.Tensor, torch.Tensor]:
    """Label each row of ``a`` / ``b`` into (chosen, rejected) by judge score.

    Ties prefer ``a``. Both inputs are ``[B, L]``.
    """
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError(f"prefer_batch expects equal [B, L] inputs, got {tuple(a.shape)} vs {tuple(b.shape)}")
    score_a = judge.score_batch(a)
    score_b = judge.score_batch(b)
    prefer_a = score_a >= score_b  # [B]
    chosen = torch.where(prefer_a.unsqueeze(-1), a, b)
    rejected = torch.where(prefer_a.unsqueeze(-1), b, a)
    return chosen, rejected
