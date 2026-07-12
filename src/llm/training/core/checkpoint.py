"""Checkpoint save/load with atomic-write semantics."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LRScheduler

from llm.training.core.config import CheckpointConfig
from llm.training.distributed import load_model_state_dict, model_state_dict


class CheckpointManager:
    """Save/load checkpoints with retention and atomic-write semantics."""

    def __init__(self, config: CheckpointConfig, rank: int, logger):
        self.config = config
        self.rank = rank
        self.logger = logger
        self.best_loss = float("inf")
        self.loaded_extra_state: dict | None = None
        self.checkpoints_saved: list[Path] = []
        if self.rank == 0:
            Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        epoch: int,
        model: DistributedDataParallel,
        optimizer: optim.Optimizer,
        scheduler: LRScheduler,
        scaler: torch.amp.GradScaler,
        loss: float,
        extra_state: dict | None = None,
        model_config: dict | None = None,
    ):
        if self.rank != 0:
            return

        model_state_to_save = model_state_dict(model)
        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "best_loss": self.best_loss,
            "model_state": model_state_to_save,
            "model_config": model_config,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
        }
        if extra_state:
            checkpoint["extra_state"] = extra_state

        # Atomic save: write to temp file then move
        def atomic_save(data, path):
            temp_path = str(path) + ".tmp"
            torch.save(data, temp_path)
            shutil.move(temp_path, path)

        if self.config.save_best and loss < self.best_loss:
            self.best_loss = loss
            checkpoint["best_loss"] = self.best_loss
            best_path = Path(self.config.checkpoint_dir) / "best.pt"
            atomic_save(checkpoint, best_path)
            self.logger.info(f"🏆 New best model saved with loss {loss:.4f}")

        latest_path = Path(self.config.checkpoint_dir) / "latest.pt"
        atomic_save(checkpoint, latest_path)

        if (epoch + 1) % self.config.save_interval == 0:
            epoch_path = Path(self.config.checkpoint_dir) / f"epoch_{epoch + 1}.pt"
            atomic_save(checkpoint, epoch_path)
            self.checkpoints_saved.append(epoch_path)
            self._cleanup_old_checkpoints()
            self.logger.debug(f"Checkpoint saved to {epoch_path}")

    def _cleanup_old_checkpoints(self):
        while len(self.checkpoints_saved) > self.config.keep_last_n:
            oldest_checkpoint = self.checkpoints_saved.pop(0)
            if oldest_checkpoint.exists():
                try:
                    oldest_checkpoint.unlink()
                    self.logger.debug(f"Removed old checkpoint: {oldest_checkpoint}")
                except OSError as e:
                    self.logger.warning(f"Could not remove old checkpoint {oldest_checkpoint}: {e}")

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: LRScheduler,
        scaler: torch.amp.GradScaler,
        device: torch.device,
    ) -> tuple[int, float]:
        if not self.config.resume_from_checkpoint:
            return 0, float("inf")

        ckp_path = self.config.resume_from_checkpoint
        if not Path(ckp_path).exists():
            self.logger.warning(f"Checkpoint file not found: {ckp_path}. Starting from scratch.")
            return 0, float("inf")

        try:
            map_location = device
            checkpoint = torch.load(ckp_path, map_location=map_location)
            load_model_state_dict(model, checkpoint["model_state"])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            if scheduler is not None and checkpoint.get("scheduler_state") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            if scaler is not None and checkpoint.get("scaler_state") is not None:
                scaler.load_state_dict(checkpoint["scaler_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint.get("best_loss", float("inf"))
            self.best_loss = best_loss
            self.loaded_extra_state = checkpoint.get("extra_state")
            self.logger.info(f"✅ Resumed training from epoch {start_epoch} using checkpoint {ckp_path}")
            return start_epoch, best_loss
        except (OSError, RuntimeError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to load checkpoint from {ckp_path}: {e}")
            self.logger.warning("Starting from scratch due to checkpoint loading error.")
            return 0, float("inf")
