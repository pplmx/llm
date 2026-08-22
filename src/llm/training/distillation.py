"""Knowledge-distillation losses (ROADMAP 13.4 / RIL DEC-055, TASK-221).

Hinton et al. 2015: train a compact student against a frozen teacher by
softening both output distributions with a temperature ``T`` and adding a
KL term to the hard-label cross-entropy:

    loss = alpha * CE(student, y)
         + (1 - alpha) * T^2 * KL(softmax(student / T) || softmax(teacher / T))

The ``T^2`` factor keeps the two terms on the same scale (the KL of soft
distributions is ~ T^-2). Only the student's parameters receive gradients —
the teacher logits are always detached.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class DistillationLoss(nn.Module):
    """Combined hard-label CE + temperature-scaled KL distillation loss.

    Args:
        temperature: Softening temperature ``T`` (> 0). Higher ``T`` yields
            softer targets and smoother gradients.
        alpha: Hard-label CE weight in ``[0, 1]``; the KL term contributes
            ``(1 - alpha)``.
        teacher: Optional frozen teacher ``nn.Module`` whose logits are used
            when ``forward`` is called without explicit ``teacher_logits``.
            Its parameters are never back-propped into.
        ignore_index: Label value (e.g. ``-100``) whose positions are excluded
            from both the CE and the KL term.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        teacher: nn.Module | None = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"DistillationLoss temperature must be > 0, got {temperature}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"DistillationLoss alpha must be in [0, 1], got {alpha}")
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.teacher = teacher
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        inputs: torch.Tensor | None = None,
        teacher_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``alpha*CE + (1-alpha)*T^2*KL`` (scalar)."""
        if teacher_logits is None:
            if self.teacher is None:
                raise ValueError(
                    "DistillationLoss has no teacher: pass teacher_logits or an inputs batch with a teacher module."
                )
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)

        sl = student_logits.reshape(-1, student_logits.size(-1)).float()
        tl = teacher_logits.reshape(-1, teacher_logits.size(-1)).float()
        lab = labels.reshape(-1)
        if sl.size(0) != tl.size(0) or sl.size(0) != lab.numel():
            raise ValueError(
                "student_logits / teacher_logits / labels must have matching token counts "
                f"got {tuple(sl.shape)} vs {tuple(tl.shape)} vs {tuple(lab.shape)}"
            )

        temp = self.temperature
        valid = lab != self.ignore_index
        if valid.any():
            ce = self.ce(sl[valid], lab[valid])
            s_logp = torch.log_softmax(sl[valid] / temp, dim=-1)
            t_p = torch.softmax(tl[valid].detach() / temp, dim=-1)
            kl = functional.kl_div(s_logp, t_p, reduction="batchmean")
        else:
            ce = torch.zeros((), device=sl.device)
            kl = torch.zeros((), device=sl.device)
        return self.alpha * ce + (1.0 - self.alpha) * (temp**2) * kl
