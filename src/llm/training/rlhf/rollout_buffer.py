"""
Rollout Buffer for PPO training.

Stores experiences from policy rollouts for PPO updates.
"""

from dataclasses import dataclass

import torch


@dataclass
class RolloutSample:
    """Single sample from a rollout."""

    prompt_ids: torch.Tensor  # [prompt_len]
    response_ids: torch.Tensor  # [response_len]
    full_input_ids: torch.Tensor  # [prompt_len + response_len]
    attention_mask: torch.Tensor  # [total_len]
    rewards: torch.Tensor  # [1] scalar reward from reward model
    old_log_probs: torch.Tensor  # [response_len] log probs from policy
    values: torch.Tensor | None = None  # [response_len] value estimates (optional)
    advantages: torch.Tensor | None = None  # [response_len] computed advantages


@dataclass
class RolloutBatch:
    """Batch of rollout samples for PPO training."""

    input_ids: torch.Tensor  # [batch_size, max_len]
    attention_mask: torch.Tensor  # [batch_size, max_len]
    response_mask: torch.Tensor  # [batch_size, max_len] mask for response tokens only
    old_log_probs: torch.Tensor  # [batch_size, max_response_len]
    rewards: torch.Tensor  # [batch_size]
    advantages: torch.Tensor  # [batch_size, max_response_len]
    returns: torch.Tensor  # [batch_size, max_response_len]


class RolloutBuffer:
    """
    Buffer to store and manage rollout experiences for PPO.

    Stores samples from policy rollouts and provides mini-batches
    for PPO updates with proper padding and masking.
    """

    def __init__(
        self,
        gae_lambda: float = 0.95,
        gamma: float = 1.0,
        normalize_advantages: bool = True,
    ):
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.normalize_advantages = normalize_advantages

        self.samples: list[RolloutSample] = []

    def add(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor,
        values: torch.Tensor | None = None,
    ):
        """Add a rollout sample to the buffer."""
        full_input_ids = torch.cat([prompt_ids, response_ids], dim=0)
        attention_mask = torch.ones(len(full_input_ids), dtype=torch.long)

        sample = RolloutSample(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            full_input_ids=full_input_ids,
            attention_mask=attention_mask,
            rewards=rewards,
            old_log_probs=old_log_probs,
            values=values,
        )
        self.samples.append(sample)

    def compute_advantages(self):
        """
        Compute advantages using GAE (Generalized Advantage Estimation).

        For language modeling with sparse rewards (reward only at end),
        we use a simplified approach where the advantage of each token
        is based on the final reward.
        """
        for sample in self.samples:
            response_len = len(sample.response_ids)
            reward = sample.rewards.item()

            if sample.values is not None:
                # Full GAE computation
                values = sample.values
                advantages = torch.zeros(response_len)
                last_gae = 0.0

                for t in reversed(range(response_len)):
                    if t == response_len - 1:
                        next_value = 0.0
                        delta = reward - values[t]
                    else:
                        next_value = values[t + 1]
                        delta = self.gamma * next_value - values[t]

                    last_gae = delta + self.gamma * self.gae_lambda * last_gae
                    advantages[t] = last_gae
            else:
                # Simplified: uniform advantage based on final reward
                advantages = torch.full((response_len,), reward / response_len)

            sample.advantages = advantages

        # Normalize advantages across all samples
        if self.normalize_advantages and len(self.samples) > 0:
            all_advantages = torch.cat([s.advantages for s in self.samples])
            mean = all_advantages.mean()
            std = all_advantages.std() + 1e-8
            for sample in self.samples:
                sample.advantages = (sample.advantages - mean) / std

    def get_batches(
        self,
        mini_batch_size: int,
        shuffle: bool = True,
        device: torch.device | str = "cpu",
    ):
        """
        Generate mini-batches for PPO updates.

        Yields:
            RolloutBatch: Padded and masked mini-batch of experiences.
        """
        if not self.samples:
            return

        # Shuffle samples
        indices = list(range(len(self.samples)))
        if shuffle:
            import random

            random.shuffle(indices)

        # Generate mini-batches
        for start_idx in range(0, len(indices), mini_batch_size):
            batch_indices = indices[start_idx : start_idx + mini_batch_size]
            batch_samples = [self.samples[i] for i in batch_indices]

            yield self._collate_batch(batch_samples, device)

    def _collate_batch(
        self,
        samples: list[RolloutSample],
        device: torch.device | str,
    ) -> RolloutBatch:
        """Collate samples into a padded batch."""
        batch_size = len(samples)

        # Find max lengths
        max_total_len = max(len(s.full_input_ids) for s in samples)
        max_response_len = max(len(s.response_ids) for s in samples)

        # Initialize tensors
        input_ids = torch.zeros(batch_size, max_total_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_total_len, dtype=torch.long)
        response_mask = torch.zeros(batch_size, max_total_len, dtype=torch.long)
        old_log_probs = torch.zeros(batch_size, max_response_len)
        advantages = torch.zeros(batch_size, max_response_len)
        returns = torch.zeros(batch_size, max_response_len)
        rewards = torch.zeros(batch_size)

        # Fill tensors
        for i, sample in enumerate(samples):
            total_len = len(sample.full_input_ids)
            prompt_len = len(sample.prompt_ids)
            response_len = len(sample.response_ids)

            input_ids[i, :total_len] = sample.full_input_ids
            attention_mask[i, :total_len] = 1
            response_mask[i, prompt_len:total_len] = 1
            old_log_probs[i, :response_len] = sample.old_log_probs
            rewards[i] = sample.rewards

            if sample.advantages is not None:
                advantages[i, :response_len] = sample.advantages
                # Returns = advantages + values (or just rewards for simplified case)
                if sample.values is not None:
                    returns[i, :response_len] = sample.advantages + sample.values
                else:
                    returns[i, :response_len] = sample.rewards

        return RolloutBatch(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            response_mask=response_mask.to(device),
            old_log_probs=old_log_probs.to(device),
            rewards=rewards.to(device),
            advantages=advantages.to(device),
            returns=returns.to(device),
        )

    def clear(self):
        """Clear all samples from the buffer."""
        self.samples.clear()

    def __len__(self) -> int:
        return len(self.samples)
