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
        # Avoid deepcopy (unsafe for some builds), use reconstruction
        self.ref_model = super().build_model()
        self.ref_model.load_state_dict(policy_model.state_dict())

        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        return policy_model

    def on_train_start(self, engine):
        """Called by engine when training starts."""
        pass

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
        log_probs = F.log_softmax(shift_logits, dim=-1)

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

        rejected_input_ids = batch["rejected_input_ids"]
        rejected_labels = batch["rejected_labels"]

        # Ensure ref_model is on correct device
        if self.ref_model is not None:
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
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = pi_logratios - ref_logratios

        losses = -F.logsigmoid(self.beta * logits)
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
