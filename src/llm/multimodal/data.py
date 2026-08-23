"""A CPU-verifiable multimodal DataModule pairing text + an auxiliary modality.

Spike (RIL TASK-226 / DEC-058): establishes the *batch + data-module contract*
for multimodal training without touching :class:`DecoderModel`. Each synthetic
sample pairs token ids (with next-token labels) with a raw auxiliary-modality
feature vector; the registry's encoder projects that feature into embeddings,
so a batch is a dict of

    ``{"input_ids": [B, T], "labels": [B, T], "modal_embeds": [B, embed_dim]}``

A future multimodal training task/model consumes ``modal_embeds`` (mapped to
per-token embeddings) — that integration is a separate slice, not part of this
data/encoder contract spike.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from llm.data.modules.map_base import SamplerMapDataModule
from llm.multimodal.encoders import MODALITY_ENCODER_REGISTRY, ModalityEncoder


class MultimodalDataModule(SamplerMapDataModule):
    """Synthetic text + modality DataModule that encodes via the registry.

    Args:
        config: the training ``Config`` (batch size / vocab / seq len / samples).
        modality: registered encoder name (default ``"linear"``).
        input_dim: raw feature dimension of the auxiliary modality.
        embed_dim: encoder output dimension (defaults to ``config.model.hidden_size``).
    """

    def __init__(
        self,
        config: Any,
        modality: str = "linear",
        input_dim: int = 16,
        embed_dim: int | None = None,
    ) -> None:
        super().__init__(config)
        self.modality = modality
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim or getattr(config.model, "hidden_size", 24))
        self.encoder: ModalityEncoder | None = None

    def build_encoder(self) -> ModalityEncoder:
        encoder_cls = MODALITY_ENCODER_REGISTRY.get(self.modality)
        return encoder_cls(self.input_dim, self.embed_dim)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        vocab = self.config.model.vocab_size
        seq_len = self.config.model.max_seq_len
        num = self.config.training.num_samples
        val_num = max(1, num // 10)

        # Fixed seed for reproducible, rank-identical synthetic data (mirrors
        # the SyntheticDataModule RIL ISS-134 pattern — never the global RNG).
        gen = torch.Generator()
        gen.manual_seed(0)
        # Generate ``num + val_num`` samples so the validation split is DISJOINT
        # from training (previously val_dataset trained on the training set, which
        # silently made val metrics equal train metrics — deep-dive TASK-228).
        rows = [
            (torch.arange(vocab).repeat(seq_len // vocab + 1)[:seq_len] + i).fmod(vocab).long()
            for i in range(num + val_num)
        ]
        inputs = torch.stack(rows)
        labels = inputs  # next-token self-supervised labels
        raw_features = torch.randn(num + val_num, self.input_dim, generator=gen)

        self.encoder = self.build_encoder()
        self.encoder.eval()
        with torch.no_grad():
            modal_embeds = self.encoder(raw_features)

        self.train_dataset = TensorDataset(inputs[:num], labels[:num], modal_embeds[:num])
        self.val_dataset = TensorDataset(inputs[num:], labels[num:], modal_embeds[num:])

    def _collate(self, batch) -> dict[str, torch.Tensor]:
        inputs, labels, embeds = zip(*batch, strict=True)
        return {
            "input_ids": torch.stack(inputs),
            "labels": torch.stack(labels),
            "modal_embeds": torch.stack(embeds),
        }

    def train_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler | None]:
        if self.train_dataset is None:
            raise ValueError("Train dataset not initialized.")
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
