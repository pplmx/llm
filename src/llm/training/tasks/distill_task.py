"""Knowledge-distillation training task (ROADMAP 13.4 / RIL DEC-055).

``DistillationTask`` trains a compact STUDENT model against a FROZEN teacher
using :class:`DistillationLoss` (``alpha*CE + (1-alpha)*T^2*KL``). It keeps the
``lm`` batch/loss contract (``(input_ids, labels)``, LM shift) so it drops into
the standard engine loop, but its ``train_step`` runs BOTH the student and the
(frozen, gradient-detached) teacher on each batch and distills the teacher's
softened distribution into the student.

The teacher is loaded from a checkpoint referenced by
``training.distill_teacher_path`` (a model trained with any ``lm``-family task
and saved by the :class:`~llm.training.core.utils.CheckpointManager`). The
teacher must share the architecture of ``config.model`` (same vocab / dims) so
its ``state_dict`` loads directly; the weights differ (the teacher is
pre-trained, the student is fresh). If ``distill_teacher_path`` is unset, a
developer/testing convenience path builds a freshly-seeded (frozen) teacher —
used by the unit tests and quick smoke runs, not by real distillation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from llm.runtime import ModelFactory
from llm.training.core.checkpoint import load_checkpoint_payload
from llm.training.distillation import DistillationLoss
from llm.training.tasks.lm_task import LanguageModelingTask

# Distinct construction seeds so the frozen teacher and the fresh student start
# from different weights (a frozen clone would make KL==0 trivially).
_TEACHER_SEED = 11111
_STUDENT_SEED = 22222


class DistillationTask(LanguageModelingTask):
    """Standard-loop LM task that distills a frozen teacher into the student."""

    def supports_pipeline_parallel(self) -> bool:
        # train_step runs BOTH student and teacher logits; the pipeline stage
        # contract is a single-model loss, so PP must be refused.
        return False

    def build_model(self) -> nn.Module:
        tcfg = self.config.training
        # The student is the model the optimizer updates; the teacher is a
        # frozen, pre-trained clone of the same architecture.
        torch.manual_seed(_STUDENT_SEED)
        student = ModelFactory.from_config(self.config.model)
        teacher = self._build_teacher()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        student.to(next(teacher.parameters()).device)
        self.teacher = teacher
        self._distill_temperature = float(tcfg.distill_temperature)
        self._distill_alpha = float(tcfg.distill_alpha)
        return student

    def _build_teacher(self) -> nn.Module:
        """Load the frozen teacher — from ``distill_teacher_path`` when set,
        else from a fresh fixed seed (dev/test convenience only)."""
        path = getattr(self.config.training, "distill_teacher_path", None)
        teacher = ModelFactory.from_config(self.config.model)
        if path is None:
            torch.manual_seed(_TEACHER_SEED)
            return ModelFactory.from_config(self.config.model)
        payload = load_checkpoint_payload(path)
        if payload is None or "model_state" not in payload:
            raise ValueError(
                f"distill_teacher_path={path!r} does not point at a loadable "
                "checkpoint (missing model_state). Train + save an lm-family "
                "model with CheckpointManager first."
            )
        state = payload["model_state"]
        # Strip any distributed-wrapping prefixes a checkpoint may carry.
        clean: dict[str, torch.Tensor] = {}
        for key, tensor in state.items():
            clean[key.removeprefix("module.").removeprefix("_orig_mod.")] = tensor
        teacher.load_state_dict(clean)
        return teacher

    def build_criterion(self) -> nn.Module:
        return DistillationLoss(
            temperature=self._distill_temperature,
            alpha=self._distill_alpha,
            teacher=self.teacher if getattr(self, "teacher", None) is not None else None,
        )

    def _shifted(self, logits: torch.Tensor, labels: torch.Tensor):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        if shift_labels.numel() == 0:
            raise ValueError("sequence length must be > 1 for next-token language modeling")
        return shift_logits, shift_labels

    def _kd_step(self, batch, model: nn.Module, criterion: nn.Module):
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            targets = batch["labels"]
        else:
            input_ids, targets = batch
        # Keep the frozen teacher on the student's device (it is not part of
        # the model the engine moves, so follow the input device explicitly).
        if next(self.teacher.parameters()).device != input_ids.device:
            self.teacher.to(input_ids.device)
        student_logits = model(input_ids)
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids)
        s_shift, labels_shift = self._shifted(student_logits, targets.to(student_logits.device))
        t_shift, _ = self._shifted(teacher_logits, targets)
        loss = criterion(
            s_shift.view(-1, s_shift.size(-1)).float(),
            labels_shift.view(-1),
            teacher_logits=t_shift.view(-1, t_shift.size(-1)).float(),
        )
        return loss

    def train_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        loss = self._kd_step(batch, model, criterion)
        if torch.isnan(loss):
            return torch.tensor(0.0, device=loss.device, requires_grad=True), {"loss": 0.0, "ppl": 1.0}
        return loss, {
            "loss": loss.item(),
            "ppl": torch.exp(loss).item() if loss.item() < 20 else float("inf"),
        }

    def validation_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        with torch.no_grad():
            loss = self._kd_step(batch, model, criterion)
        return loss, {
            "val_loss": loss.item(),
            "val_ppl": torch.exp(loss).item() if loss.item() < 20 else float("inf"),
        }
