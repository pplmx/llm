import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

from llm.models.decoder import DecoderModel
from llm.training.tasks.base_task import TrainingTask


class LanguageModelingTask(TrainingTask):
    """
    A task for causal language modeling.
    """

    def build_model(self) -> nn.Module:
        model_config = self.config.model
        # We assume vocab_size is provided in the model_config or derived elsewhere.
        # For now, let's assume it's in model_config.
        # If not, it might need to be passed during task initialization.
        vocab_size = getattr(model_config, "vocab_size", 50257)  # Default GPT-2 vocab size

        return DecoderModel(
            vocab_size=vocab_size,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads if hasattr(model_config, "num_heads") else 8,
            intermediate_size=model_config.intermediate_size,
            embedding_dropout_p=model_config.dropout,
            attn_dropout_p=model_config.dropout,
            mlp_dropout_p=model_config.dropout,
            use_moe=model_config.use_moe,
            num_experts=model_config.num_experts,
            top_k=model_config.top_k,
            # New architectural params
            num_kv_heads=model_config.num_kv_heads,
            use_glu=model_config.use_glu,
            norm_type=getattr(
                model_config, "norm_type", nn.LayerNorm
            ),  # Still relying on strict type or manual map? Config usually has strings.
            # Convert string norm_impl to class if we had a registry for norms.
            # For now DecoderModel accepts type or module. Config has no way to pass type class directly safely from YAML.
            # We'll ignore norm_impl for now or handle it later.
            max_seq_len=model_config.max_seq_len,
            attn_impl=model_config.attn_impl,
            mlp_impl=model_config.mlp_impl,
        )

    def build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        # Filter parameters that require gradients
        param_groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "weight_decay": self.config.training.weight_decay,
            }
        ]

        return optim.AdamW(
            param_groups,
            lr=self.config.training.lr,
            eps=1e-8,
            betas=(0.9, 0.95),
            fused=torch.cuda.is_available(),
        )

    def build_scheduler(self, optimizer: optim.Optimizer) -> LRScheduler | None:
        if self.config.training.scheduler_type == "cosine":
            main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.config.training.epochs - self.config.training.warmup_epochs),
                eta_min=self.config.training.lr * 0.1,
            )
        else:
            # Default to step or none if not specified/supported
            return None

        if self.config.training.warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-6, end_factor=1.0, total_iters=self.config.training.warmup_epochs
            )
            return optim.lr_scheduler.SequentialLR(
                optimizer,
                [warmup_scheduler, main_scheduler],
                milestones=[self.config.training.warmup_epochs],
            )

        return main_scheduler

    def build_criterion(self) -> nn.Module:
        # Cross Entropy Loss for next token prediction
        return nn.CrossEntropyLoss()

    def train_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        # Batch is expected to be (input_ids, labels) or dict
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            targets = batch["labels"]
        else:
            input_ids, targets = batch

        logits = model(input_ids)

        # Reshape for cross entropy: (batch * seq_len, vocab_size)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        if torch.isnan(loss):
            return torch.tensor(0.0, device=loss.device, requires_grad=True), {"loss": 0.0, "ppl": 1.0}

        metrics = {
            "loss": loss.item(),
            "ppl": torch.exp(loss).item() if loss.item() < 20 else float("inf"),  # PPL calculation
        }

        return loss, metrics

    def validation_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            targets = batch["labels"]
        else:
            input_ids, targets = batch

        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        metrics = {"val_loss": loss.item(), "val_ppl": torch.exp(loss).item() if loss.item() < 20 else float("inf")}

        return loss, metrics
