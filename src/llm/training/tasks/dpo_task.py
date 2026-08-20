from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from llm.training.tasks.lm_task import LanguageModelingTask

# Key under which the frozen reference weights live in checkpoint extra_state.
# The DPO reference is the "base" the policy is penalised against diverging
# from (normally the SFT model). It is snapshot inside ``build_model`` — from
# the *initial* policy — which runs BEFORE the engine loads any checkpoint
# into the policy, so without persisting the reference a resumed DPO run would
# compute every log-ratio against a freshly-random model (RIL round-60
# deep-dive Finding 1).
REF_MODEL_STATE_KEY = "dpo_ref_model"


class DPOTask(LanguageModelingTask):
    """
    Task for Direct Preference Optimization (DPO).
    """

    def __init__(self, config: Any, data_module: Any):
        super().__init__(config, data_module)
        self.ref_model: nn.Module | None = None
        # Beta parameter for DPO, default 0.1
        self.beta = getattr(config.training, "dpo_beta", 0.1)
        self._ref_from_checkpoint = False

    def build_model(self) -> nn.Module:
        policy_model = super().build_model()

        # Create reference model
        # Avoid deepcopy (unsafe for some builds), use reconstruction
        self.ref_model = super().build_model()
        self.ref_model.load_state_dict(policy_model.state_dict())

        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        return policy_model

    def get_checkpoint_state(self) -> dict[str, Any] | None:
        """Persist the frozen reference so a resumed DPO run keeps the ORIGINAL
        base model (RIL round-60 deep-dive Finding 1).

        ``build_model`` snapshots the reference *before* the engine loads a
        checkpoint into the policy and the checkpoint never carried the
        reference, so a resumed run otherwise computes log-ratios against a
        random model. Snapshot to CPU so the pickled sidecar is device-neutral.
        """
        if self.ref_model is None:
            return None
        return {
            REF_MODEL_STATE_KEY: {
                key: value.detach().to("cpu").clone() for key, value in self.ref_model.state_dict().items()
            }
        }

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        """Restore the frozen reference from a DPO checkpoint's extra_state.

        Marks the reference as checkpoint-restored; :meth:`on_checkpoint_loaded`
        then keeps this ORIGINAL base instead of re-syncing to the (now-moved)
        policy when resuming a mid-DPO checkpoint.
        """
        if not state or self.ref_model is None:
            return
        ref_state = state.get(REF_MODEL_STATE_KEY)
        if ref_state:
            self.ref_model.load_state_dict(ref_state)
            self._ref_from_checkpoint = True

    def on_checkpoint_loaded(self, model: nn.Module) -> None:
        """Align the frozen reference with the checkpoint-loaded base policy.

        Called by the engine right after ``load_checkpoint`` applies the
        resumed weights to the policy. Two cases:

        - Resuming from an SFT/base checkpoint (no persisted DPO reference in
          extra_state): the reference must equal the loaded policy — that IS
          the base the policy diverges from. ``_ref_from_checkpoint`` stays
          False so we sync here.
        - Resuming from a mid-DPO checkpoint: ``load_checkpoint_state`` already
          restored the ORIGINAL base reference; keep it (do NOT overwrite with
          the moved policy, which would silently corrupt every later log-ratio).
        """
        if self.ref_model is None:
            return
        if self._ref_from_checkpoint:
            return
        self.ref_model.load_state_dict(model.state_dict())

    def _get_batch_logps(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        average_log_prob: bool = False,
    ) -> torch.Tensor:
        """
        Compute log probability of the labels given the logits.
        """
        if logits.shape[1] != labels.shape[1]:
            raise ValueError(f"Logits seq_len {logits.shape[1]} != Labels seq_len {labels.shape[1]}")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute log_softmax
        log_probs = functional.log_softmax(shift_logits, dim=-1)

        # Gather log probs of the labels
        # Create mask where labels != -100
        mask = (shift_labels != -100).float()

        temp_labels = shift_labels.clone()
        temp_labels[temp_labels == -100] = 0
        selected_log_probs = torch.gather(log_probs, dim=-1, index=temp_labels.unsqueeze(-1)).squeeze(-1)

        selected_log_probs = selected_log_probs * mask

        # Sum over sequence
        sum_log_probs = selected_log_probs.sum(dim=1)

        if average_log_prob:
            divisor = mask.sum(dim=1)
            return sum_log_probs / (divisor + 1e-8)

        return sum_log_probs

    def train_step(self, batch: dict[str, Any], model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        chosen_input_ids = batch["chosen_input_ids"]
        chosen_labels = batch["chosen_labels"]
        chosen_attention_mask = batch["chosen_attention_mask"]

        rejected_input_ids = batch["rejected_input_ids"]
        rejected_labels = batch["rejected_labels"]
        rejected_attention_mask = batch["rejected_attention_mask"]

        # Ensure ref_model is on correct device
        if self.ref_model is not None:
            model_device = next(model.parameters()).device
            ref_device = next(self.ref_model.parameters()).device

            if ref_device != model_device:
                self.ref_model = self.ref_model.to(model_device)

        # Policy Forward. The data pipeline builds per-row attention masks
        # (chosen/rejected_attention_mask) that SFT and Reward pass to the
        # model — DPO discarded them, so the causal model attended to the
        # padding tokens (a right-padded batch leaks pad-token scores into
        # every valid position's context), diverging from SFT/Reward semantics
        # (RIL ISS-249).
        policy_chosen_logits = model(chosen_input_ids, attn_mask=chosen_attention_mask)
        policy_rejected_logits = model(rejected_input_ids, attn_mask=rejected_attention_mask)

        policy_chosen_logps = self._get_batch_logps(policy_chosen_logits, chosen_labels)
        policy_rejected_logps = self._get_batch_logps(policy_rejected_logits, rejected_labels)

        if self.ref_model is None:
            raise RuntimeError("reference model was not built")
        # Reference Forward (No Grad) — same masks so the reference sees the
        # identical context the policy sees.
        with torch.no_grad():
            ref_chosen_logits = self.ref_model(chosen_input_ids, attn_mask=chosen_attention_mask)
            ref_rejected_logits = self.ref_model(rejected_input_ids, attn_mask=rejected_attention_mask)

            ref_chosen_logps = self._get_batch_logps(ref_chosen_logits, chosen_labels)
            ref_rejected_logps = self._get_batch_logps(ref_rejected_logits, rejected_labels)

        # DPO Loss
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = pi_logratios - ref_logratios

        losses = -functional.logsigmoid(self.beta * logits)
        loss = losses.mean()

        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps).detach()
        reward_acc = (chosen_rewards > rejected_rewards).float().mean()

        metrics = {
            "loss": loss.item(),
            "reward_chosen": chosen_rewards.mean().item(),
            "reward_rejected": rejected_rewards.mean().item(),
            "reward_acc": reward_acc.item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        }

        return loss, metrics

    def validation_step(
        self, batch: dict[str, Any], model: nn.Module, criterion: nn.Module
    ) -> tuple[torch.Tensor, dict]:
        loss, metrics = self.train_step(batch, model, criterion)
        val_metrics = {f"val_{k}": v for k, v in metrics.items()}
        return loss, val_metrics
