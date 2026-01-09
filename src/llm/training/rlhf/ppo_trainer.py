"""
PPO Trainer for RLHF.

Implements Proximal Policy Optimization for language model alignment.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from llm.training.rlhf.config import PPOConfig
from llm.training.rlhf.rollout_buffer import RolloutBatch, RolloutBuffer

logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for RLHF.

    Trains a policy model using PPO with rewards from a reward model.
    Optionally uses a reference model for KL divergence penalty.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: nn.Module,
        tokenizer: Any,
        config: PPOConfig,
        ref_model: nn.Module | None = None,
        value_model: nn.Module | None = None,
        device: str | torch.device = "cuda",
    ):
        """
        Initialize PPO trainer.

        Args:
            policy_model: The language model to train (policy).
            reward_model: Frozen reward model for scoring responses.
            tokenizer: Tokenizer for encoding/decoding.
            config: PPO configuration.
            ref_model: Frozen reference model for KL penalty (optional).
            value_model: Value function model (optional, uses policy if None).
            device: Device to run training on.
        """
        self.policy = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(device)

        # Reference model (frozen copy of initial policy)
        if config.use_ref_model:
            if ref_model is not None:
                self.ref_model = ref_model
            else:
                # Create a frozen copy
                self.ref_model = self._create_ref_model()
        else:
            self.ref_model = None

        # Value model (shares backbone with policy if not provided)
        self.value_model = value_model

        # Move models to device
        self.policy.to(self.device)
        self.reward_model.to(self.device)
        self.reward_model.eval()
        if self.ref_model is not None:
            self.ref_model.to(self.device)
            self.ref_model.eval()
        if self.value_model is not None:
            self.value_model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=config.policy_lr or 1e-5,
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            gae_lambda=config.gae_lambda,
            gamma=config.gamma,
            normalize_advantages=config.normalize_advantages,
        )

        # Training stats
        self.global_step = 0
        self.kl_ctl = config.kl_coef

    def _create_ref_model(self) -> nn.Module:
        """Create a frozen copy of the policy model."""
        import copy

        ref_model = copy.deepcopy(self.policy)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model

    def generate_responses(
        self,
        prompts: list[str],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Generate responses for a batch of prompts.

        Returns:
            prompt_ids: List of prompt token tensors
            response_ids: List of response token tensors
            log_probs: List of log probability tensors for responses
        """
        self.policy.eval()

        all_prompt_ids = []
        all_response_ids = []
        all_log_probs = []

        with torch.no_grad():
            for prompt in prompts:
                prompt_ids = torch.tensor(
                    self.tokenizer.encode(prompt),
                    dtype=torch.long,
                    device=self.device,
                )

                # Generate response autoregressively
                response_ids = []
                log_probs = []

                input_ids = prompt_ids.unsqueeze(0)  # [1, prompt_len]

                for _ in range(self.config.response_max_len):
                    logits = self.policy(input_ids)  # [1, seq_len, vocab_size]
                    next_token_logits = logits[0, -1, :]  # [vocab_size]

                    # Apply temperature
                    if self.config.temperature != 1.0:
                        next_token_logits = next_token_logits / self.config.temperature

                    # Sample next token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    # Get log probability
                    log_prob = F.log_softmax(next_token_logits, dim=-1)[next_token]

                    response_ids.append(next_token.item())
                    log_probs.append(log_prob.item())

                    # Update input
                    input_ids = torch.cat(
                        [input_ids, next_token.unsqueeze(0)],
                        dim=1,
                    )

                    # Check for EOS (simplified: check for common EOS tokens)
                    if hasattr(self.tokenizer, "eos_id") and next_token.item() == self.tokenizer.eos_id:
                        break

                all_prompt_ids.append(prompt_ids)
                all_response_ids.append(torch.tensor(response_ids, device=self.device))
                all_log_probs.append(torch.tensor(log_probs, device=self.device))

        self.policy.train()
        return all_prompt_ids, all_response_ids, all_log_probs

    def compute_rewards(
        self,
        prompt_ids: list[torch.Tensor],
        response_ids: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Compute rewards for prompt-response pairs using the reward model.
        """
        rewards = []

        with torch.no_grad():
            for p_ids, r_ids in zip(prompt_ids, response_ids, strict=True):
                # Concatenate prompt and response
                full_ids = torch.cat([p_ids, r_ids]).unsqueeze(0)  # [1, total_len]
                attention_mask = torch.ones_like(full_ids)

                # Get reward
                reward = self.reward_model(full_ids, attention_mask)  # [1]
                rewards.append(reward.squeeze())

        return rewards

    def compute_kl_penalty(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between policy and reference model.
        """
        if self.ref_model is None:
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            ref_logits = self.ref_model(input_ids)

        policy_logits = self.policy(input_ids)

        # Compute KL divergence on response tokens only
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)

        # KL = sum(p * (log p - log q))
        kl = (policy_log_probs.exp() * (policy_log_probs - ref_log_probs)).sum(dim=-1)

        # Mask to response tokens only
        kl = (kl * response_mask).sum() / response_mask.sum().clamp(min=1)

        return kl

    def ppo_step(self, batch: RolloutBatch) -> dict[str, float]:
        """
        Perform a single PPO update step.

        Returns:
            Dictionary of training metrics.
        """
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        response_mask = batch.response_mask
        old_log_probs = batch.old_log_probs
        advantages = batch.advantages
        returns = batch.returns  # noqa: F841 - prepared for value function updates

        # Forward pass
        logits = self.policy(input_ids)

        # Compute new log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Get log probs for actual tokens (shifted)
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_response_mask = response_mask[:, 1:]  # noqa: F841 - prepared for masked loss

        new_log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            new_log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        # Compute ratio
        # Note: old_log_probs needs to be aligned with token positions
        response_len = old_log_probs.size(1)
        # Extract log probs for response portion only
        batch_size = input_ids.size(0)
        new_response_log_probs = torch.zeros_like(old_log_probs)

        for i in range(batch_size):
            prompt_len = (1 - response_mask[i]).sum().long()
            resp_len = response_mask[i].sum().long()
            if resp_len > 0:
                new_response_log_probs[i, :resp_len] = token_log_probs[i, prompt_len - 1 : prompt_len - 1 + resp_len]

        ratio = (new_response_log_probs - old_log_probs).exp()

        # Clipped surrogate objective
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        )

        # Mask advantages to response length
        response_advantages = advantages[:, :response_len]

        policy_loss_1 = -response_advantages * ratio
        policy_loss_2 = -response_advantages * clipped_ratio
        policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

        # KL penalty
        kl = self.compute_kl_penalty(input_ids, attention_mask, response_mask)
        kl_loss = self.kl_ctl * kl

        # Total loss
        loss = policy_loss + kl_loss

        # Entropy bonus (optional)
        if self.config.entropy_coef > 0:
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            entropy = (entropy * response_mask[:, :-1]).sum() / response_mask[:, :-1].sum().clamp(min=1)
            loss = loss - self.config.entropy_coef * entropy
        else:
            entropy = torch.tensor(0.0)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.max_grad_norm,
            )

        self.optimizer.step()

        # Metrics
        with torch.no_grad():
            approx_kl = ((ratio - 1) - (ratio.log())).mean()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "kl": kl.item(),
            "kl_loss": kl_loss.item(),
            "entropy": entropy.item() if isinstance(entropy, torch.Tensor) else entropy,
            "approx_kl": approx_kl.item(),
            "ratio_mean": ratio.mean().item(),
        }

    def train_step(self, prompts: list[str]) -> dict[str, float]:
        """
        Perform a complete RLHF training step.

        1. Generate responses for prompts
        2. Compute rewards
        3. Store in buffer and compute advantages
        4. Perform PPO updates for multiple epochs

        Args:
            prompts: List of prompt strings.

        Returns:
            Dictionary of training metrics.
        """
        # 1. Generate responses
        prompt_ids, response_ids, log_probs = self.generate_responses(prompts)

        # 2. Compute rewards
        rewards = self.compute_rewards(prompt_ids, response_ids)

        # 3. Store in buffer
        self.buffer.clear()
        for p_ids, r_ids, lp, reward in zip(prompt_ids, response_ids, log_probs, rewards, strict=True):
            self.buffer.add(
                prompt_ids=p_ids,
                response_ids=r_ids,
                rewards=reward,
                old_log_probs=lp,
            )

        # 4. Compute advantages
        self.buffer.compute_advantages()

        # 5. PPO epochs
        all_metrics = []
        for _epoch in range(self.config.ppo_epochs):
            for batch in self.buffer.get_batches(
                mini_batch_size=self.config.mini_batch_size,
                device=self.device,
            ):
                metrics = self.ppo_step(batch)
                all_metrics.append(metrics)

                # Early stopping based on KL
                if self.config.target_kl is not None and metrics["approx_kl"] > self.config.target_kl:
                    logger.info(f"Early stopping: KL {metrics['approx_kl']:.4f} > target {self.config.target_kl}")
                    break

        self.global_step += 1

        # Aggregate metrics
        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

        avg_metrics["reward_mean"] = sum(r.item() for r in rewards) / len(rewards)
        avg_metrics["response_len_mean"] = sum(len(r) for r in response_ids) / len(response_ids)

        return avg_metrics
