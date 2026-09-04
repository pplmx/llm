"""GRPO reward-scored prompt/response groups (TASK-229, CPU-verifiable)."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from llm.data.modules.map_base import SamplerMapDataModule


class GRPODataModule(SamplerMapDataModule):
    """Synthetic GRPO groups: ``G`` responses per prompt with a 0/1 reward.

    Each group's **first** response (offset 0 within the group) is the target —
    ONE fixed random sequence (seeded, drawn from the same uniform range as the
    distractors) — scored 1.0; the others are random and scored 0.0. The policy
    learns to make the target the most-likely response in its group, so the
    group-reward fraction (argmax == target) rises from ≈ chance to ~1 with
    training — a CPU-verifiable demonstration of GRPO's group-relative objective.

    The target must be a *fixed random* sequence rather than an all-zeros (or
    short-pattern) response: under the (correct) shifted next-token metric a
    fresh random model already assigns the shortest/lowest-id sequences a
    systematically higher summed log-prob, so such a target starts saturated and
    GRPO's improvement is unobservable. A fixed random target is genuinely
    in-distribution: a random model cannot tell it from the distractors (RIL
    ISS-333).

    The batch is exactly ``batch_size`` responses (one full epoch batch) with
    ``batch_size`` a multiple of ``group_size`` and stable (unshuffled) ordering,
    so the task can snapshot the initial-policy log-probs once and keep them as
    the GRPO ``old`` reference across the run.

    Like the rest of the module this is single-process only (see
    :meth:`_require_single_process`).
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.group_size = int(getattr(config.training, "grpo_group_size", 4))
        batch_size = int(config.training.batch_size)
        if batch_size % self.group_size != 0:
            raise ValueError(
                f"GRPO batch_size ({batch_size}) must be a multiple of grpo_group_size ({self.group_size})"
            )
        self.num_groups = batch_size // self.group_size

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        batch_size = int(self.config.training.batch_size)
        seq_len = int(self.config.model.max_seq_len)
        vocab = int(self.config.model.vocab_size)
        if vocab < 2:
            raise ValueError("GRPO vocab_size must be >= 2")

        # Deterministic draw: a seeded generator for the distractors AND a
        # separate seeded generator for the target, so the target is one fixed
        # random sequence from the SAME uniform range (in-distribution).
        gen = torch.Generator()
        gen.manual_seed(0)
        responses = torch.randint(1, vocab, (batch_size, seq_len), generator=gen)
        target_gen = torch.Generator()
        target_gen.manual_seed(0)
        target = torch.randint(1, vocab, (seq_len,), generator=target_gen)
        # Mark each group's first response as the target (fixed random seq, reward 1).
        for group in range(self.num_groups):
            responses[group * self.group_size] = target
        rewards = torch.zeros((batch_size,), dtype=torch.float)
        rewards[:: self.group_size] = 1.0

        self.response_tokens = responses
        self.group_rewards = rewards.reshape(-1, self.group_size)
        labels = responses  # per-token targets == the response (next-token LM gather)
        self.train_dataset = TensorDataset(responses, labels, rewards)
        self.val_dataset = self.train_dataset

    def _collate(self, batch) -> dict[str, torch.Tensor]:
        inputs, labels, rewards = zip(*batch, strict=True)
        return {
            "input_ids": torch.stack(inputs),
            "labels": torch.stack(labels),
            "rewards": torch.stack(rewards),
        }

    @staticmethod
    def _require_single_process(world_size: int) -> None:
        """Refuse distributed GRPO.

        ``DistributedSampler(shuffle=False)`` hands each rank the *interleaved*
        indices (``rank, rank+ws, ...``), so a rank's slice is not a set of whole
        groups: ``group_advantages(rewards.reshape(-1, group_size))`` in
        :class:`GRPOTask` would silently mix responses from different prompts
        (and can crash when the per-rank total is not a multiple of
        ``group_size``). The "single stable batch" + initial-policy ``old``
        snapshot premise also only holds at world_size=1 (RIL ISS-333).
        """
        if world_size > 1:
            raise ValueError(
                f"GRPODataModule requires world_size=1 (got world_size={world_size}): the "
                f"stable single group batch and the initial-policy 'old' snapshot do not "
                f"compose with a DistributedSampler, which interleaves indices across ranks "
                f"and mixes responses from different groups. Run GRPO single-process."
            )

    def train_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler | None]:
        if self.train_dataset is None:
            raise ValueError("GRPO train dataset not initialized.")
        self._require_single_process(world_size)
        # Stable (unshuffled) ordering so the single-batch GRPO "old" snapshot stays valid.
        sampler = DistributedSampler(
            self.train_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
        )
        return self.build_dataloader(self.train_dataset, sampler, collate_fn=self._collate), sampler

    def val_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader | None, DistributedSampler | None]:
        if self.train_dataset is None:
            return None, None
        self._require_single_process(world_size)
        sampler = DistributedSampler(
            self.train_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        return self.build_dataloader(self.train_dataset, sampler, collate_fn=self._collate), sampler
