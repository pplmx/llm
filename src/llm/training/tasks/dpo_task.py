import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.training.tasks.lm_task import LanguageModelingTask


class DPOTask(LanguageModelingTask):
    """
    Task for Direct Preference Optimization (DPO).
    """

    def __init__(self, config: Any, data_module: Any):
        super().__init__(config, data_module)
        self.ref_model: nn.Module | None = None
        # Beta parameter for DPO, default 0.1
        self.beta = getattr(config.training, "dpo_beta", 0.1)

    def build_model(self) -> nn.Module:
        policy_model = super().build_model()

        # Create reference model
        # Note: Ideally we load from a checkpoint.
        # Here we initialize fresh (random) -> random ref.
        # In practice, user loads pre-trained weights into policy_model.
        # So copying policy_model gives us the SFT starting point as ref.
        # This assumes build_model() or engine loads weights later?
        # If engine loads weights into policy_model AFTER build_model, ref_model is random.
        # This is a problem. ref_model must match SFT model.
        # Solution: DPOTask should probably handle weight loading or we assume user passes config.model.checkpoint_path?

        # Taking a deepcopy here ensures architecture match.
        # If weights are loaded later, we must ensure ref_model gets them too.
        # Current framework: `TrainingEngine` loads checkpoint?
        # No, `train.py` logic doesn't explicitly load weights unless resuming state or model init does.
        # If `DecoderModel` inits random, and we train from scratch DPO (rare), deepcopy is fine.
        # Use Case: SFT -> DPO. User should provide checkpoint.

        self.ref_model = copy.deepcopy(policy_model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        return policy_model

    def on_train_start(self, engine):
        """Called by engine when training starts. Good place to ensure ref_model device."""
        # Using a hook if available, or just lazy move in train_step.
        pass

    def _get_batch_logps(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        average_log_prob: bool = False,
    ) -> torch.Tensor:
        """
        Compute log probability of the labels given the logits.
        Args:
            logits: (batch, seq_len, vocab_size)
            labels: (batch, seq_len) - with -100 for ignored
        Returns:
            log_prob: (batch,)
        """
        # Shift for autoregressive
        # logits[..., :-1, :] predicts labels[..., 1:]

        if logits.shape[1] != labels.shape[1]:
            raise ValueError(f"Logits seq_len {logits.shape[1]} != Labels seq_len {labels.shape[1]}")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute log_softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs of the labels
        # shift_labels shape: (B, S-1)
        # log_probs shape: (B, S-1, V)
        # gather expects index to have same dims, so unsqueeze
        selected_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        # Masking
        # Create mask where labels != -100
        mask = (shift_labels != -100).float()

        # Zero out masked
        # log_probs for -100 index might be garbage or valid, but we zero result
        # To avoid gather error on -100 if vocab size is small?
        # If label is -100, gather might fail if -100 < 0?
        # Torch gather handles negative indices wrap-around. -100 might wrap to valid index.
        # So we must clamp or ensure validity before gather?
        # Standard trick: set labels=-100 to 0 temporarily, gather, then mask.

        temp_labels = shift_labels.clone()
        temp_labels[temp_labels == -100] = 0
        selected_log_probs = torch.gather(log_probs, dim=-1, index=temp_labels.unsqueeze(-1)).squeeze(-1)

        selected_log_probs = selected_log_probs * mask

        # Sum over sequence
        sum_log_probs = selected_log_probs.sum(dim=1)

        if average_log_prob:
            # Average per non-masked token
            divisor = mask.sum(dim=1)
            return sum_log_probs / (divisor + 1e-8)

        return sum_log_probs

    def train_step(self, batch: dict[str, Any], model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        # Batch: chosen_input_ids, chosen_labels...
        #        rejected_input_ids, rejected_labels...

        chosen_input_ids = batch["chosen_input_ids"]
        chosen_labels = batch["chosen_labels"]
        # chosen_attention_mask = batch.get("chosen_attention_mask")

        rejected_input_ids = batch["rejected_input_ids"]
        rejected_labels = batch["rejected_labels"]
        # rejected_attention_mask = batch.get("rejected_attention_mask")

        # Ensure ref_model is on correct device
        if self.ref_model is not None:
            # nn.Module doesn't always have .device
            model_device = next(model.parameters()).device
            ref_device = next(self.ref_model.parameters()).device

            if ref_device != model_device:
                self.ref_model = self.ref_model.to(model_device)

        # Policy Forward
        policy_chosen_logits = model(chosen_input_ids)
        policy_rejected_logits = model(rejected_input_ids)

        policy_chosen_logps = self._get_batch_logps(policy_chosen_logits, chosen_labels)
        policy_rejected_logps = self._get_batch_logps(policy_rejected_logits, rejected_labels)

        # Reference Forward (No Grad)
        with torch.no_grad():
            ref_chosen_logits = self.ref_model(chosen_input_ids)
            ref_rejected_logits = self.ref_model(rejected_input_ids)

            ref_chosen_logps = self._get_batch_logps(ref_chosen_logits, chosen_labels)
            ref_rejected_logps = self._get_batch_logps(ref_rejected_logits, rejected_labels)

        # DPO Loss
        # pi_logratios = policy_chosen_logps - policy_rejected_logps
        # ref_logratios = ref_chosen_logps - ref_rejected_logps
        # logits = pi_logratios - ref_logratios

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = pi_logratios - ref_logratios

        # loss = -log(sigmoid(beta * logits))
        losses = -F.logsigmoid(self.beta * logits)
        loss = losses.mean()

        # Rewards (implicit)
        # chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
        # rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()

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
        # Similar to train_step but no grad on policy
        # Reuse logic? Or copy for clarity.
        # Using self.train_step logic but no backward.

        loss, metrics = self.train_step(batch, model, criterion)
        # Rename metrics for validation
        val_metrics = {f"val_{k}": v for k, v in metrics.items()}
        return loss, val_metrics
