"""GRPO training task (TASK-229, ROADMAP 阶段十一)."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as functional

from llm.models.decoder import DecoderModel
from llm.runtime import ModelFactory
from llm.training.rlhf.grpo import GRPOLoss, group_advantages
from llm.training.tasks.lm_task import LanguageModelingTask

#: Extra-state key under which the frozen GRPO KL reference is persisted, so a
#: resumed run keeps the ORIGINAL base policy instead of a fresh random model
#: (mirrors DPO's ``REF_MODEL_STATE_KEY``).
GRPO_REF_MODEL_STATE_KEY = "grpo_ref_model"


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
        self._ref_from_checkpoint = False

    def supports_pipeline_parallel(self) -> bool:
        return False

    def build_model(self) -> nn.Module:
        policy = ModelFactory.from_config(self.config.model)

        # GRPO KL reference = the INITIAL policy (the base it diverges from) —
        # NOT a second random network. ``ModelFactory.from_config`` is called
        # at a different RNG point than the policy, so a naive second build
        # produced a random model and the K3 penalty measured divergence from
        # noise instead of policy drift. Build a matching model and COPY the
        # policy's weights, then freeze (mirrors DPO's ``build_model``).
        ref = cast(DecoderModel, ModelFactory.from_config(self.config.model))
        ref.load_state_dict(policy.state_dict())
        ref.eval()
        for param in ref.parameters():
            param.requires_grad_(False)
        self._reference = ref

        return policy

    def build_criterion(self) -> nn.Module:
        return GRPOLoss(
            clip_eps=float(self.config.training.grpo_clip_eps),
            kl_beta=float(self.config.training.grpo_kl_beta),
        )

    def _build_reference(self, device: torch.device) -> DecoderModel:
        if self._reference is None:
            # build_model not called (e.g. a bench path) — fall back to a fresh
            # policy so a stale/random reference is never silently used.
            self._reference = cast(
                DecoderModel,
                ModelFactory.from_config(self.config.model),
            )
            for param in self._reference.parameters():
                param.requires_grad_(False)
            self._reference.eval()
        return self._reference.to(device)

    @staticmethod
    def _per_token_log_probs(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Per-token next-token log-prob of the response tokens (teacher-forcing).

        Shifted factorization: ``logits[t]`` predicts ``ids[t+1]``, so gather
        ``logits[..., :-1]`` at ``ids[..., 1:]`` — matching the repo's canonical
        :meth:`LanguageModelingTask._get_batch_logps`. Gathering ``ids[t]`` from
        ``logits[t]`` instead lets the model inflate the objective by copying
        input tokens (its own embedding leaks through the residual stream) and
        included a spurious ``t=0`` "self-prediction" term (RIL ISS-333).
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_ids = ids[..., 1:].contiguous()
        if shift_ids.numel() == 0:
            raise ValueError("sequence length must be > 1 for next-token log-probs")
        return functional.log_softmax(shift_logits, dim=-1).gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)

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

    def get_checkpoint_state(self) -> dict[str, Any] | None:
        """Persist the frozen KL reference so a resumed GRPO run keeps the
        ORIGINAL base policy (mirrors DPO's round-60 fix).

        ``build_model`` snapshots the reference *before* the engine loads a
        checkpoint into the policy, so without this a resumed run would compute
        the KL penalty against a random model. Snapshot to CPU so the pickled
        sidecar is device-neutral.
        """
        if self._reference is None:
            return None
        return {
            GRPO_REF_MODEL_STATE_KEY: {
                key: value.detach().to("cpu").clone() for key, value in self._reference.state_dict().items()
            }
        }

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        """Restore the frozen reference from a GRPO checkpoint's extra_state."""
        if not state or self._reference is None:
            return
        ref_state = state.get(GRPO_REF_MODEL_STATE_KEY)
        if ref_state:
            self._reference.load_state_dict(ref_state)
            self._ref_from_checkpoint = True

    def on_checkpoint_loaded(self, model: nn.Module) -> None:
        """Align the KL reference with the checkpoint-loaded base policy.

        Called by the engine right after ``load_checkpoint`` applies the resumed
        weights to the policy:

        - Resuming from a base/SFT checkpoint (no persisted GRPO reference):
          the reference must equal the loaded policy — that IS the base. Sync here.
        - Resuming from a mid-GRPO checkpoint: ``load_checkpoint_state`` restored
          the ORIGINAL base reference; keep it rather than overwriting with the
          (moved) policy, which would corrupt every later KL penalty.
        """
        if self._reference is None or self._ref_from_checkpoint:
            return
        self._reference.load_state_dict(model.state_dict())

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
