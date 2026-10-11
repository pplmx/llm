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
    # RIL ISS-178: whether the episode was cut off by ``response_max_len``
    # (as opposed to genuinely terminating at EOS). Truncated episodes are
    # NOT terminal — GAE must bootstrap from the critic's value at the
    # post-response state ``s_L`` instead of hardcoding next_value=0.
    truncated: bool = False
    # Critic value V(s_L) at the state after the last response token, used
    # to bootstrap truncated episodes. Scalar tensor; only meaningful when
    # ``truncated`` is True and a critic is in use.
    terminal_value: torch.Tensor | None = None
    advantages: torch.Tensor | None = None  # [response_len] computed advantages
    returns: torch.Tensor | None = None  # [response_len] raw returns (advantage + value)


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


def _gae_advantages(
    values: torch.Tensor,
    reward: torch.Tensor,
    *,
    terminal_value: torch.Tensor | None,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """Vectorized GAE for one sparse-reward response.

    For a response of length ``L`` the per-token advantage is the first-order
    recurrence ``A_t = delta_t + c * A_{t+1}`` with ``c = gamma * gae_lambda``,
    ``A_L = 0`` and

    .. math::

        \\delta_t = r_t + \\gamma V_{t+1} - V_t

    where the sparse reward ``r_t`` is non-zero only at the last response token
    (``r_L = reward``) and ``V_L`` is the terminal bootstrap (``terminal_value``
    for a truncated episode, else ``0`` at EOS). Expanding the recurrence gives
    the closed form ``A_t = sum_{k=0}^{L-1-t} c^k * delta_{t+k}``, evaluated in
    O(L) vectorized torch (no Python loop, no per-element device syncs) via the
    identity ``A_{L-1-i} = c^i * S[i]`` with
    ``S[i] = cumsum_{m=0..i}( rev_delta[m] / c^m )``.

    ``values`` is detached: the GAE signal must never backprop through the
    critic's value estimate (the original Python loop's ``.item()`` reads made
    this implicit).
    """
    values = values.detach()
    device = values.device
    dtype = values.dtype
    n = values.shape[0]
    if n == 0:
        return torch.zeros(0, device=device, dtype=dtype)

    # Sparse reward: only the last response token carries the terminal reward.
    r_t = torch.zeros(n, device=device, dtype=dtype)
    r_t[-1] = reward.reshape(())
    next_value_last = (
        terminal_value.reshape(()).to(device=device, dtype=dtype)
        if terminal_value is not None
        else torch.zeros((), device=device, dtype=dtype)
    )

    next_values = torch.zeros_like(values)
    next_values[:-1] = values[1:]
    next_values[-1] = next_value_last

    delta = r_t + gamma * next_values - values
    c = gamma * gae_lambda
    if c == 0:
        # No advantage accumulation (gamma==0 or gae_lambda==0): A_t = delta_t.
        return delta

    arange = torch.arange(n, device=device)
    # The closed form A_{n-1-i} = c^i * cumsum(rev_delta/c^m) divides by c^m,
    # which overflows float32 for small c (e.g. tiny gae_lambda). Upcast the
    # reduction to float64, which covers every realistic config with a huge
    # margin, then cast back to the values' dtype.
    c_power = torch.pow(torch.full((n,), c, device=device, dtype=torch.float64), arange)
    buf = torch.cumsum(torch.flip(delta, dims=[0]).to(torch.float64) / c_power, dim=0) * c_power
    advantages = torch.flip(buf, dims=[0]).to(dtype=dtype)

    if not torch.isfinite(advantages).all():
        # Pathological config (|c| so close to 0 that 1/c^n exceeds float64's
        # range): fall back to the stable forward scan so the result is exact
        # for ANY input, matching the original loop.
        acc = torch.zeros((), device=device, dtype=dtype)
        for t in range(n - 1, -1, -1):
            acc = delta[t] + c * acc
            advantages[t] = acc
    return advantages


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
        truncated: bool = False,
        terminal_value: torch.Tensor | None = None,
    ):
        """Add a rollout sample to the buffer.

        Args:
            truncated: True when the response hit ``response_max_len``
                instead of terminating at EOS — such episodes are not
                genuinely terminal and GAE bootstraps from ``terminal_value``.
            terminal_value: Critic estimate V(s_L) at the state after the
                last response token (only used for truncated episodes).
        """
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
            truncated=truncated,
            terminal_value=terminal_value,
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
                # Full GAE with sparse terminal reward.
                values = sample.values
                response_len = len(sample.response_ids)
                # RIL ISS-178: the episode's last state is terminal ONLY when
                # the response ended at EOS — then nothing follows and
                # next_value is 0. A response truncated by
                # ``response_max_len`` is NOT finished: the policy would keep
                # generating, so the last state must be bootstrapped from the
                # critic's value V(s_L) at the post-response state (decayed by
                # ``gamma`` like any bootstrap). Hardcoding 0 modeled both
                # cases identically and systematically deflated the final-token
                # advantages of every truncated rollout.
                terminal = sample.terminal_value if sample.truncated else None
                advantages = _gae_advantages(
                    values,
                    sample.rewards,
                    terminal_value=terminal,
                    gamma=self.gamma,
                    gae_lambda=self.gae_lambda,
                )
                # Raw returns (advantage + value estimate) — the critic's
                # regression target. These must be computed from the raw
                # advantage, *before* normalization, so the value function
                # learns the true return scale.  Using the normalized
                # advantage here (V + (A-mean)/std) trains the critic on a
                # shifted/scaled target while the next rollout's GAE
                # bootstrap consumes the critic's output *raw* — the two
                # sides then disagree on scale and corrupt the advantage
                # signal every rollout.
                sample.returns = advantages.clone() + values
            else:
                device = sample.rewards.device
                # Simplified: uniform advantage based on final reward
                advantages = torch.full(
                    (response_len,),
                    reward / response_len,
                    device=device,
                )

            sample.advantages = advantages

        # Normalize advantages across all samples
        if self.normalize_advantages and len(self.samples) > 0:
            all_advantages = torch.cat([s.advantages for s in self.samples if s.advantages is not None])
            if all_advantages.numel() == 0:
                raise RuntimeError("no advantages computed for normalization")
            mean = all_advantages.mean()
            std = all_advantages.std() + 1e-8
            for sample in self.samples:
                if sample.advantages is None:
                    raise RuntimeError("advantages were not computed")
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
                if sample.returns is not None:
                    # Raw (unnormalized) returns — see compute_advantages.
                    returns[i, :response_len] = sample.returns
                elif sample.values is not None:
                    # Backwards compatibility: components that construct
                    # RolloutSample without calling compute_advantages.
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
