"""Rejection-sampling DataModule: emit the kept (high-reward) responses as a CE set."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from llm.data.modules.map_base import SamplerMapDataModule
from llm.training.rlhf.rejection_sampling import rejection_sample


class RejectionSampleDataModule(SamplerMapDataModule):
    """Synthetic rejection-sampling DataModule (TASK-230 / ROADMAP 11.4).

    Generates ``num_samples`` responses whose reward is 1 iff the response ends
    in token 0 (else 0), applies the chosen rejection filter (top-K or threshold),
    and exposes the **kept** responses as an SFT-style next-token dataset.
    """

    def __init__(
        self,
        config: Any,
        *,
        mode: str = "top_k",
        k: int | None = None,
        threshold: float | None = None,
    ) -> None:
        super().__init__(config)
        if mode not in ("top_k", "threshold"):
            raise ValueError(f"mode must be 'top_k' or 'threshold', got {mode!r}")
        self.mode = mode
        self.k = k
        self.threshold = threshold
        self.stats: dict[str, float] | None = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        batch = int(self.config.training.num_samples)
        seq_len = int(self.config.model.max_seq_len)
        vocab = int(self.config.model.vocab_size)
        if vocab < 2:
            raise ValueError("RejectionSampleDataModule requires vocab_size >= 2")

        gen = torch.Generator()
        gen.manual_seed(0)
        responses = torch.randint(1, vocab, (batch, seq_len), generator=gen)
        ends_with_target = torch.rand(batch, generator=gen) < 0.5
        responses[ends_with_target, -1] = 0
        rewards = (responses[:, -1] == 0).float()

        if self.mode == "top_k":
            k_value = self.k
            if k_value is None:
                raise ValueError("RejectionSampleDataModule mode='top_k' requires k")
            mask, stats = rejection_sample(rewards, k=k_value)
        else:
            threshold_value = self.threshold
            if threshold_value is None:
                raise ValueError("RejectionSampleDataModule mode='threshold' requires threshold")
            mask, stats = rejection_sample(rewards, threshold=threshold_value)
        kept = responses[mask]
        self.response_tokens = responses
        self.rewards = rewards
        self.kept_mask = mask
        self.stats = stats
        self.train_dataset = TensorDataset(kept, kept)  # labels == inputs (next-token)
        self.val_dataset = self.train_dataset

    def _collate(self, batch) -> dict[str, torch.Tensor]:
        inputs, labels = zip(*batch, strict=True)
        return {"input_ids": torch.stack(inputs), "labels": torch.stack(labels)}

    def train_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler | None]:
        if self.train_dataset is None:
            raise ValueError("Rejection-sampling train dataset not initialized.")
        sampler = DistributedSampler(
            self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
        )
        return self.build_dataloader(self.train_dataset, sampler, collate_fn=self._collate), sampler

    def val_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader | None, DistributedSampler | None]:
        if self.train_dataset is None:
            return None, None
        sampler = DistributedSampler(
            self.train_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        return self.build_dataloader(self.train_dataset, sampler, collate_fn=self._collate), sampler
