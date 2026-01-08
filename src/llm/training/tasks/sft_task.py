from typing import Any

import torch
import torch.nn as nn

from llm.training.tasks.lm_task import LanguageModelingTask


class SFTTask(LanguageModelingTask):
    """
    Task for Supervised Fine-Tuning (SFT).
    Inherits from LanguageModelingTask but handles dictionary batches and attention masks correctly.
    """

    def train_step(self, batch: dict[str, Any], model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attn_mask = batch.get("attention_mask")

        # Forward pass
        # Note: DecoderModel expects attn_mask to be a boolean/float mask usually,
        # or it generates causal mask internally.
        # If we provide attn_mask, it's combined.
        # Check DecoderModel signature.
        logits = model(input_ids, attn_mask=attn_mask)

        # Shift logits and labels for next token prediction
        # (B, S, V) -> (B, S, V)
        # Standard causal LM loss: logits[..., :-1, :] predicts labels[..., 1:]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if torch.isnan(loss):
            return torch.tensor(0.0, device=loss.device, requires_grad=True), {"loss": 0.0}

        metrics = {
            "loss": loss.item(),
            "ppl": torch.exp(loss).item() if loss.item() < 20 else float("inf"),
        }

        return loss, metrics

    def validation_step(
        self, batch: dict[str, Any], model: nn.Module, criterion: nn.Module
    ) -> tuple[torch.Tensor, dict]:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attn_mask = batch.get("attention_mask")

        logits = model(input_ids, attn_mask=attn_mask)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        metrics = {"val_loss": loss.item(), "val_ppl": torch.exp(loss).item() if loss.item() < 20 else float("inf")}

        return loss, metrics

    def build_criterion(self) -> nn.Module:
        # Cross Entropy Loss with ignore_index=-100 (standard)
        # We ensure it explicitly uses -100, though default is -100.
        return nn.CrossEntropyLoss(ignore_index=-100)
