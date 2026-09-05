"""Multimodal training task (TASK-227, ADR-013)."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn

from llm.models.decoder import DecoderModel
from llm.multimodal.data import MultimodalDataModule
from llm.multimodal.encoders import ModalityEncoder
from llm.multimodal.model import MultimodalModel
from llm.runtime import ModelFactory
from llm.training.tasks.lm_task import LanguageModelingTask


class MultimodalTask(LanguageModelingTask):
    """Standard-loop LM task whose model consumes a modality prefix.

    The paired :class:`~llm.multimodal.data.MultimodalDataModule` supplies
    ``{"input_ids", "labels", "modal_embeds"}`` batches; the model is a
    :class:`MultimodalModel` that conditions text next-token prediction on the
    merged modal+text sequence.
    """

    def supports_pipeline_parallel(self) -> bool:
        # train_step feeds both input_ids and modal_embeds; the pipeline stage
        # contract is a single-model loss, so PP must be refused.
        return False

    def build_model(self) -> nn.Module:
        decoder = cast(DecoderModel, ModelFactory.from_config(self.config.model))
        # Trainable-tower mode (ROADMAP 12.1 slice 2): the vision encoder is
        # owned by the model so its parameters train jointly with the decoder;
        # the default precompute path keeps encoder=None and consumes frozen
        # ``modal_embeds`` (backward compatible).
        encoder: ModalityEncoder | None = None
        if getattr(self.data_module, "train_encoder", False):
            encoder = cast(MultimodalDataModule, self.data_module).build_encoder()
        # The modal prefix width is an encoder property (``encoder.num_tokens``
        # from the input geometry), not a config choice; the old
        # ``model.multimodal_modal_tokens`` knob was stored but never read by
        # forward/generate (RIL TASK-311/ISS-349) and is removed.
        return MultimodalModel(decoder, encoder=encoder)

    def build_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss(ignore_index=-100)

    def _multimodal_loss(self, batch, model: nn.Module, criterion: nn.Module) -> torch.Tensor:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        if "modal_samples" in batch:
            # Trainable-tower path: encode raw images/audio inside the model
            # forward so gradients reach the modal encoder.
            logits = model(input_ids, modal_samples=batch["modal_samples"])
        else:
            logits = model(input_ids, modal_embeds=batch["modal_embeds"])
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        if shift_labels.numel() == 0:
            raise ValueError("sequence length must be > 1 for next-token language modeling")
        return criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    def train_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        loss = self._multimodal_loss(batch, model, criterion)
        return loss, {
            "loss": loss.item(),
            "ppl": math.exp(loss.item()) if loss.item() < 20 else float("inf"),
        }

    def validation_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        with torch.no_grad():
            loss = self._multimodal_loss(batch, model, criterion)
        return loss, {
            "val_loss": loss.item(),
            "val_ppl": math.exp(loss.item()) if loss.item() < 20 else float("inf"),
        }
