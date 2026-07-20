import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

from llm.core.adalora import apply_adalora
from llm.core.adapter import apply_adapter
from llm.core.bitfit import apply_bitfit
from llm.core.ia3 import apply_ia3
from llm.core.peft import apply_peft
from llm.core.prefix_tuning import apply_prefix_tuning
from llm.runtime import ModelFactory
from llm.training.core.callbacks import AdaLoRAPruningCallback
from llm.training.tasks.base_task import TrainingTask


class LanguageModelingTask(TrainingTask):
    """
    A task for causal language modeling.
    """

    def build_model(self) -> nn.Module:
        model = ModelFactory.from_config(self.config.model)
        t_cfg = self.config.training
        # New unified PEFT dispatch (T2 PEFT #44). When ``peft_method``
        # is set, the method is resolved through ``PEFT_REGISTRY`` and
        # ``peft_kwargs`` is forwarded verbatim to ``apply_peft`` (and
        # thence to the per-method ``apply_*`` function). This is the
        # recommended path going forward — the legacy ``use_*`` flag
        # branches below are preserved for backward compatibility with
        # existing configs (which set the flags directly without going
        # through the registry).
        if getattr(t_cfg, "peft_method", None) is not None:
            kwargs = dict(t_cfg.peft_kwargs or {})
            apply_peft(model, t_cfg.peft_method, **kwargs)
            return model
        # Legacy per-method flag path (T3 #40-#42, T2 PEFT #18-#20).
        # Each ``use_*`` flag stays opt-in; defaults preserve current
        # behavior so existing configs are unaffected. The five branches
        # below are intentional — when ``peft_method`` is None we still
        # need them. New PEFT methods should land on the new path
        # (``peft_method`` + ``peft_kwargs``), not by appending here.
        # AdaLoRA is opt-in. When ``use_adalora=True`` we wrap every
        # ``nn.Linear`` (or the filtered subset) in ``AdaLoRALinear``
        # and freeze the base weights. The pruning callback registered
        # by ``build_callbacks`` then drives the rank schedule.
        if getattr(t_cfg, "use_adalora", False):
            apply_adalora(
                model,
                init_rank=t_cfg.adalora_init_rank,
                target_rank=t_cfg.adalora_target_rank,
                alpha=t_cfg.adalora_alpha,
                target_modules=t_cfg.adalora_target_modules,
                orth_reg_weight=t_cfg.adalora_orth_reg_weight,
            )
        # Prefix Tuning is opt-in. When ``use_prefix_tuning=True`` we
        # wrap every ``MultiHeadAttention`` (or the filtered subset) in
        # ``PrefixTuningAttention`` and freeze the base MHA so only the
        # prefix path is trainable. Unlike AdaLoRA there is no
        # scheduler / tracker — the user calls
        # ``fold_reparameterization`` at inference time (matching the
        # LoRA apply / merge pattern).
        if getattr(t_cfg, "use_prefix_tuning", False):
            apply_prefix_tuning(
                model,
                prefix_len=t_cfg.prefix_tuning_len,
                reparam_hidden=t_cfg.prefix_reparam_hidden,
                target_modules=t_cfg.prefix_target_modules,
            )
        # IA³ is opt-in. When ``use_ia3=True`` we wrap every
        # ``nn.Linear`` (or the filtered subset) in ``IA3Linear`` and
        # freeze the base weight so only ``ia3_l`` is trainable. Like
        # Prefix Tuning there is no scheduler / tracker — the user
        # calls ``merge_ia3`` at inference time (matching the LoRA
        # apply / merge pattern).
        if getattr(t_cfg, "use_ia3", False):
            apply_ia3(
                model,
                init_scale=t_cfg.ia3_init_scale,
                target_modules=t_cfg.ia3_target_modules,
            )
        # BitFit is opt-in. When ``use_bitfit=True`` we freeze every
        # parameter and enable gradients on every bias (or the
        # filtered subset). Like Prefix Tuning and IA³ there is no
        # scheduler / tracker — BitFit is a one-shot
        # ``requires_grad`` toggle at ``build_model`` time, with no
        # inference-time merge step (the biases are simply left in
        # place at inference — they cost nothing extra).
        if getattr(t_cfg, "use_bitfit", False):
            apply_bitfit(
                model,
                target_modules=t_cfg.bitfit_target_modules,
            )
        # Adapter Layers (Houlsby 2019) is opt-in. When
        # ``use_adapter=True`` we wrap every ``nn.Linear`` (or the
        # filtered subset) in ``AdapterLinear`` (down → activation →
        # up bottleneck residual) and freeze the base weight so only
        # the adapter is trainable. Like Prefix Tuning / IA³ / BitFit
        # there is no scheduler / tracker — ``apply_adapter`` is a
        # one-shot wrap at ``build_model`` time, and there is no
        # inference-time merge (the up projection is zero, so the
        # wrapper contributes nothing unless trained).
        if getattr(t_cfg, "use_adapter", False):
            apply_adapter(
                model,
                bottleneck_dim=t_cfg.adapter_bottleneck_dim,
                target_modules=t_cfg.adapter_target_modules,
            )
        return model

    def build_callbacks(self) -> list:
        """Register the AdaLoRA pruning callback when ``use_adalora=True``
        and the PEFT adapter-checkpoint callback when ``peft_method``
        is set.

        The callbacks are wired after ``build_model`` runs, so any
        tracker / state they need to construct sees the model in its
        final position (and any DDP/FSDP wrapper that the engine
        applies on top).
        """
        callbacks: list = []
        t_cfg = self.config.training

        # T3 #42 AdaLoRA pruning callback (periodic prune-to-rank
        # cadence driven by the optimizer-step counter).
        if getattr(t_cfg, "use_adalora", False):
            callbacks.append(
                AdaLoRAPruningCallback(
                    use_adalora=True,
                    adalora_init_rank=t_cfg.adalora_init_rank,
                    adalora_target_rank=t_cfg.adalora_target_rank,
                    adalora_ema_alpha=t_cfg.adalora_ema_alpha,
                    adalora_tinit=t_cfg.adalora_tinit,
                    adalora_tfinal=t_cfg.adalora_tfinal,
                    adalora_prune_every=t_cfg.adalora_prune_every,
                )
            )

        # T2 PEFT #48 PEFT adapter sidecar (one-shot write at
        # on_train_end via save_peft). The default path is
        # ``{checkpoint_dir}/peft_adapter_{method}.bin`` when no
        # explicit ``peft_save_path`` is set — the method-name suffix
        # avoids clobbering when the user later switches PEFT methods.
        peft_method = getattr(t_cfg, "peft_method", None)
        if peft_method is not None:
            from pathlib import Path

            from llm.training.core.callbacks import (
                PEFTAdapterCheckpointCallback,
            )

            explicit_path = getattr(t_cfg, "peft_save_path", None)
            if explicit_path is not None:
                resolved_path: Path | None = Path(explicit_path)
            else:
                ckpt_dir = getattr(
                    getattr(self.config, "checkpoint", None),
                    "checkpoint_dir",
                    None,
                )
                if ckpt_dir is None:
                    resolved_path = None
                else:
                    resolved_path = (
                        Path(ckpt_dir) / f"peft_adapter_{peft_method}.bin"
                    )

            callbacks.append(
                PEFTAdapterCheckpointCallback(
                    peft_method=peft_method,
                    peft_kwargs=dict(t_cfg.peft_kwargs or {}),
                    peft_save_path=resolved_path,
                )
            )

        return callbacks

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
