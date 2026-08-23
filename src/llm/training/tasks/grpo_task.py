"""GRPO training task (TASK-229, ROADMAP 阶段十一)."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as functional

from llm.models.decoder import DecoderModel
from llm.runtime import ModelFactory
from llm.training.rlhf.grpo import GRPOLoss, group_advantages
from llm.training.tasks.lm_task import LanguageModelingTask


class GRPOTask(LanguageModelingTask):
    """Standard-loop task that optimizes a policy with GRPO group advantages.

    The paired :class:`~llm.data.modules.grpo.GRPODataModule` yields a single
    stable batch of ``batch_size`` responses (``group_size`` per prompt). The
    task snapshots the policy's initial log-probs as the GRPO ``old`` reference,
    keeps a frozen reference policy for the KL penalty, and applies
    :class:`GRPOLoss` with group z-score advantages computed from the rewards.
    """

    def __init__(self, config, data_module) -> None:
        super().__init__(config, data_module)
        self._reference: DecoderModel | None = None
        self.group_size = int(getattr(config.training, "grpo_group_size", 4))
        self._old_log_probs: torch.Tensor | None = None

    def supports_pipeline_parallel(self) -> bool:
        return False

    def build_model(self) -> nn.Module:
        return ModelFactory.from_config(self.config.model)

    def build_criterion(self) -> nn.Module:
        return GRPOLoss(
            clip_eps=float(self.config.training.grpo_clip_eps),
            kl_beta=float(self.config.training.grpo_kl_beta),
        )

    def _build_reference(self, device: torch.device) -> DecoderModel:
        if self._reference is None:
            ref = cast(DecoderModel, ModelFactory.from_config(self.config.model))
            for param in ref.parameters():
                param.requires_grad_(False)
            ref.eval()
            self._reference = ref
        return self._reference.to(device)

    @staticmethod
    def _per_token_log_probs(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Per-token next-token log-prob of the actual response tokens (teacher-forcing)."""
        return functional.log_softmax(logits, dim=-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)

    def _grpo_loss(self, batch, model: nn.Module, criterion: nn.Module) -> torch.Tensor:
        input_ids = batch["input_ids"]
        rewards = batch["rewards"]
        device = input_ids.device

        logits = model(input_ids)
        lp = self._per_token_log_probs(logits, input_ids)

        with torch.no_grad():
            ref_logits = self._build_reference(device)(input_ids)
            ref_lp = self._per_token_log_probs(ref_logits, input_ids)
        if (
            self._old_log_probs is None
            or self._old_log_probs.shape != lp.shape
            or self._old_log_probs.device != lp.device
        ):
            self._old_log_probs = lp.detach()  # initial policy snapshot (importance "old")

        advantages = group_advantages(rewards.reshape(-1, self.group_size)).reshape(-1)
        return criterion(lp, self._old_log_probs, ref_lp, advantages)

    def train_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        loss = self._grpo_loss(batch, model, criterion)
        return loss, {
            "grpo_loss": loss.item(),
            "reward": batch["rewards"].float().mean().item(),
        }

    def validation_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        with torch.no_grad():
            loss = self._grpo_loss(batch, model, criterion)
        return loss, {
            "val_loss": loss.item(),
            "val_reward": batch["rewards"].float().mean().item(),
        }
