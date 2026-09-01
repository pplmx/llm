"""SimPO task — Simple Preference Optimization with a Reference-Free Reward.

SimPO (Meng et al., 2024, *SimPO: Simple Preference Optimization with a
Reference-Free Reward*, arXiv:2405.14734) replaces DPO's reference-model KL
penalty with a **length-normalized, reference-free** implicit reward::

    r(x, y) = beta * mean_t log pi_theta(y_t | x, y_<t)

and optimizes a logistic target over the preferred-vs-rejected reward margin
plus a chosen-response SFT regularizer::

    L = -log sigma( r(x, y_w) - r(x, y_l) - gamma ) - lambda * mean_t log pi_theta(y_w)

Because the reward is a function of the policy itself (no frozen reference),
SimPO is cheaper and simpler than DPO and is robust to the reference-model
divergence traps that DPO must checkpoint around (RIL round-60 Finding 1).

This task reuses the same preference ``DPODataModule`` (chosen/rejected pairs)
and the standard language-modelling loop; it is fully CPU-verifiable with a
synthetic directional e2e.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from llm.training.tasks.lm_task import LanguageModelingTask


class SimPOTask(LanguageModelingTask):
    """SimPO: reference-free, length-normalized preference optimization."""

    def __init__(self, config: Any, data_module: Any):
        super().__init__(config, data_module)
        self.beta = getattr(config.training, "simpo_beta", 2.0)
        self.gamma = getattr(config.training, "simpo_gamma", 0.0)
        self.lmbda = getattr(config.training, "simpo_lambda", 1.0)

    def train_step(self, batch: dict[str, Any], model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        chosen_input_ids = batch["chosen_input_ids"]
        chosen_labels = batch["chosen_labels"]
        chosen_attention_mask = batch["chosen_attention_mask"]
        rejected_input_ids = batch["rejected_input_ids"]
        rejected_labels = batch["rejected_labels"]
        rejected_attention_mask = batch["rejected_attention_mask"]

        policy_chosen_logits = model(chosen_input_ids, attn_mask=chosen_attention_mask)
        policy_rejected_logits = model(rejected_input_ids, attn_mask=rejected_attention_mask)

        # Length-normalized (mean) log-probs -> the SimPO implicit reward.
        mean_chosen = self._get_batch_logps(policy_chosen_logits, chosen_labels, average_log_prob=True)
        mean_rejected = self._get_batch_logps(policy_rejected_logits, rejected_labels, average_log_prob=True)

        implicit_w = self.beta * mean_chosen
        implicit_l = self.beta * mean_rejected
        margin = implicit_w - implicit_l - self.gamma
        loss_rl = (-functional.logsigmoid(margin)).mean()
        # SFT regularization: maximize the chosen response's mean log-prob.
        loss_sft = -mean_chosen.mean()
        loss = loss_rl + self.lmbda * loss_sft

        reward_acc = (implicit_w > implicit_l).float().mean()
        metrics = {
            "loss": loss.item(),
            "reward_chosen": implicit_w.detach().mean().item(),
            "reward_rejected": implicit_l.detach().mean().item(),
            "reward_acc": reward_acc.item(),
            "reward_margin": (implicit_w - implicit_l).detach().mean().item(),
        }
        return loss, metrics

    def validation_step(
        self, batch: dict[str, Any], model: nn.Module, criterion: nn.Module
    ) -> tuple[torch.Tensor, dict]:
        loss, metrics = self.train_step(batch, model, criterion)
        return loss, {f"val_{k}": v for k, v in metrics.items()}
