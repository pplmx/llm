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
from llm.multimodal.audio import AudioSpectrogramEncoder
from llm.multimodal.encoders import MODALITY_ENCODER_REGISTRY, ModalityEncoder
from llm.multimodal.vision import VisionTransformerEncoder


class MultimodalDataModule(SamplerMapDataModule):
    """Synthetic text + modality DataModule that encodes via the registry.

    Args:
        config: the training ``Config`` (batch size / vocab / seq len / samples).
        modality: registered encoder name (default ``"linear"``); ``"vit"`` uses
            the CLIP/SigLIP-style vision encoder on raw images (ROADMAP 12.1).
        input_dim: raw feature dimension of the auxiliary modality (linear path).
        embed_dim: encoder output dimension (defaults to ``config.model.hidden_size``).
        image_h/image_w: image spatial size (``"vit"`` path; must be divisible by
            ``patch_size``).
        audio_frames/audio_mels: log-mel spectrogram shape (``"audio"`` path;
            must be divisible by ``patch_size``).
        patch_size: vision/audio patch side length.
        vit_layers/vit_heads: tower depth / attention heads (vision/audio paths).
        with_cls: prepend a ``[CLS]`` token in the vision/audio tower.
        train_encoder: when True the image paths emit **raw modal samples**
            (``[B, C, H, W]`` under the ``modal_samples`` batch key) and the
            model encodes them in-forward (tower trainable — image-text
            alignment, ROADMAP 12.1 slice 2); when False (default CLIP-style)
            it precomputes frozen tower features at setup.
        vit_instruction_len: when > 0, Visual Instruction Tuning semantics: each
            sample becomes ``[instruction | response]`` with the instruction
            tokens labelled ``-100`` so the CE loss only supervises the
            response, conditioned on the modal prefix + instruction context
            (ROADMAP 12.1 Visual Instruction Tuning).
    """

    def __init__(
        self,
        config: Any,
        modality: str = "linear",
        input_dim: int = 16,
        embed_dim: int | None = None,
        image_h: int = 64,
        image_w: int = 64,
        audio_frames: int = 32,
        audio_mels: int = 64,
        patch_size: int = 16,
        vit_layers: int = 4,
        vit_heads: int = 4,
        with_cls: bool = True,
        train_encoder: bool = False,
        vit_instruction_len: int = 0,
    ) -> None:
        super().__init__(config)
        self.modality = modality
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim or getattr(config.model, "hidden_size", 24))
        self.image_h = int(image_h)
        self.image_w = int(image_w)
        self.audio_frames = int(audio_frames)
        self.audio_mels = int(audio_mels)
        self.patch_size = int(patch_size)
        self.vit_layers = int(vit_layers)
        self.vit_heads = int(vit_heads)
        self.with_cls = bool(with_cls)
        self.train_encoder = bool(train_encoder)
        self.vit_instruction_len = int(vit_instruction_len)
        self._raw_modality = self.modality in ("vit", "audio")
        self.encoder: ModalityEncoder | None = None
        self.num_modal_tokens: int | None = None

    def _make_raw_samples(self, count: int, gen: torch.Generator) -> torch.Tensor | None:
        """Synthetic raw modal samples (``[B, C, H, W]``-layout) or None.

        ``"vit"`` -> RGB images, ``"audio"`` -> 1-channel log-mel spectrograms,
        ``"linear"`` -> None (the feature-vector path precomputes embeds).
        """
        if self.modality == "vit":
            return torch.randn(count, 3, self.image_h, self.image_w, generator=gen)
        if self.modality == "audio":
            return torch.randn(count, 1, self.audio_frames, self.audio_mels, generator=gen)
        return None

    def build_encoder(self) -> ModalityEncoder:
        if self.modality == "vit":
            encoder = VisionTransformerEncoder(
                embed_dim=self.embed_dim,
                layers=self.vit_layers,
                num_heads=self.vit_heads,
                patch_size=self.patch_size,
                image_h=self.image_h,
                image_w=self.image_w,
                with_cls=self.with_cls,
                # Frozen tower = CLIP/Whisper-style precompute path (features
                # read once at setup); ``train_encoder=True`` swaps to a
                # trainable tower so the model can jointly align modality+text.
                freeze_encoder=not self.train_encoder,
            )
        elif self.modality == "audio":
            encoder = AudioSpectrogramEncoder(
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                layers=self.vit_layers,
                num_heads=self.vit_heads,
                n_frames=self.audio_frames,
                n_mels=self.audio_mels,
                with_cls=self.with_cls,
                freeze_encoder=not self.train_encoder,
            )
        else:
            encoder_cls = MODALITY_ENCODER_REGISTRY.get(self.modality)
            encoder = encoder_cls(self.input_dim, self.embed_dim)
        self.num_modal_tokens = encoder.num_tokens if self.modality in ("vit", "audio") else None
        return encoder

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
        if self.vit_instruction_len > 0:
            # Visual Instruction Tuning: ``[instruction | response]`` with the
            # instruction positions masked (-100) so the shift-based LM loss in
            # the multimodal task only supervises the response, conditioned on
            # the image prefix + instruction context. Response keeps the
            # per-sample cyclic pattern (deterministic -> learnable), exactly
            # like the plain self-supervised rows otherwise.
            inst_len = min(self.vit_instruction_len, seq_len - 1)  # keep >=1 response token
            inst = torch.randint(1, vocab, (num + val_num, inst_len), generator=gen)
            resp = inputs[:, : seq_len - inst_len]
            inputs = torch.cat([inst, resp], dim=1)
            labels = torch.cat(
                [torch.full_like(inst, -100), resp],
                dim=1,
            )
        else:
            labels = inputs  # next-token self-supervised labels

        raw = self._make_raw_samples(num + val_num, gen)
        if raw is not None and self.train_encoder:
            # Trainable-tower path: batch carries RAW modal samples under the
            # ``modal_samples`` key; the model encodes them in-forward so
            # gradients flow through the tower (image/audio-text alignment).
            # The encoder is still built (owned by the model) but nothing is
            # precomputed at setup.
            self.encoder = self.build_encoder()
            self.train_dataset = TensorDataset(inputs[:num], labels[:num], raw[:num])
            self.val_dataset = TensorDataset(inputs[num:], labels[num:], raw[num:])
        else:
            self.encoder = self.build_encoder()
            self.encoder.eval()
            with torch.no_grad():
                if raw is not None:
                    # Raw samples -> image/audio-token embeddings [B, num_tokens, embed_dim]
                    modal_embeds = self.encoder(raw)
                else:
                    raw_features = torch.randn(num + val_num, self.input_dim, generator=gen)
                    modal_embeds = self.encoder(raw_features)

            self.train_dataset = TensorDataset(inputs[:num], labels[:num], modal_embeds[:num])
            self.val_dataset = TensorDataset(inputs[num:], labels[num:], modal_embeds[num:])

    def _collate(self, batch) -> dict[str, torch.Tensor]:
        if self._raw_modality and self.train_encoder:
            inputs, labels, modal_samples = zip(*batch, strict=True)
            return {
                "input_ids": torch.stack(inputs),
                "labels": torch.stack(labels),
                "modal_samples": torch.stack(modal_samples),
            }
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
