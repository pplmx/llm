"""AI-feedback (judge-labeled) preference DataModule feeding DPO (TASK-231)."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from llm.data.modules.map_base import SamplerMapDataModule
from llm.training.rlhf.aifeedback import PreferenceJudge, TargetTokenJudge, prefer_batch


class AIFeedbackDataModule(SamplerMapDataModule):
    """Synthetic response pairs labeled by a :class:`PreferenceJudge`.

    Each pair has exactly one "good" response (ends in the judge's target token),
    so the judge deterministically labels chosen/rejected. The module yields
    DPO-compatible batches:
    ``{chosen_input_ids, chosen_labels, chosen_attention_mask, rejected_* }``.
    """

    def __init__(self, config: Any, judge: PreferenceJudge | None = None) -> None:
        super().__init__(config)
        self.judge = judge or TargetTokenJudge(target_token=0)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        batch = int(self.config.training.num_samples)
        seq_len = int(self.config.model.max_seq_len)
        vocab = int(self.config.model.vocab_size)
        target = self.judge.target_token if isinstance(self.judge, TargetTokenJudge) else 0
        if vocab < 2:
            raise ValueError("AIFeedbackDataModule requires vocab_size >= 2")

        gen = torch.Generator()
        gen.manual_seed(0)
        a = torch.randint(1, vocab, (batch, seq_len), generator=gen)
        b = torch.randint(1, vocab, (batch, seq_len), generator=gen)
        # First half of rows: a is the good response; second half: b is good.
        half = batch // 2
        a[:half, -1] = target
        b[half:, -1] = target

        chosen, rejected = prefer_batch(a, b, self.judge)
        self.chosen = chosen
        self.rejected = rejected
        ones = torch.ones_like(chosen)
        self.train_dataset = TensorDataset(chosen, rejected, ones)
        self.val_dataset = self.train_dataset

    def _collate(self, batch) -> dict[str, torch.Tensor]:
        chosen, rejected, ones = zip(*batch, strict=True)
        chosen = torch.stack(chosen)
        rejected = torch.stack(rejected)
        mask = torch.stack(ones)
        return {
            "chosen_input_ids": chosen,
            "chosen_labels": chosen,
            "chosen_attention_mask": mask,
            "rejected_input_ids": rejected,
            "rejected_labels": rejected,
            "rejected_attention_mask": mask,
        }

    def train_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler | None]:
        if self.train_dataset is None:
            raise ValueError("AIFeedback train dataset not initialized.")
        sampler = DistributedSampler(
            self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        return self.build_dataloader(self.train_dataset, sampler, collate_fn=self._collate), sampler

    def val_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader | None, DistributedSampler | None]:
        if self.train_dataset is None:
            return None, None
        sampler = DistributedSampler(
            self.train_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        return self.build_dataloader(self.train_dataset, sampler, collate_fn=self._collate), sampler
