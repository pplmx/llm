from pathlib import Path  # Added Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter  # Added SummaryWriter


# Forward declaration to avoid circular imports
class TrainingEngine:
    pass


class Callback:
    """
    Base class for creating callbacks in the training engine.

    Callbacks are functions that the training engine will call at various points
    during training (e.g., at the start of training, at the end of an epoch, etc.).
    """

    def __init__(self):
        self.engine: TrainingEngine | None = None

    def set_engine(self, engine: TrainingEngine):
        """Called by the TrainingEngine to provide a reference to itself."""
        self.engine = engine

    def on_train_start(self, logs: dict[str, Any] | None = None):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, logs: dict[str, Any] | None = None):
        """Called at the end of training."""
        pass

    def on_epoch_start(self, epoch: int, logs: dict[str, Any] | None = None):
        """Called at the beginning of an epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None):
        """Called at the end of an epoch."""
        pass

    def on_batch_start(self, epoch: int, batch_idx: int, logs: dict[str, Any] | None = None):
        """Called at the beginning of a batch."""
        pass

    def on_batch_end(self, epoch: int, batch_idx: int, logs: dict[str, Any] | None = None):
        """Called at the end of a batch."""
        pass

    def on_train_step_end(
        self,
        epoch: int,
        batch_idx: int,
        loss: torch.Tensor,
        metrics: dict[str, Any],
        logs: dict[str, Any] | None = None,
    ):
        """Called after the train_step (forward/backward/optimizer.step) is completed for a batch."""
        pass

    def on_validation_start(self, epoch: int, logs: dict[str, Any] | None = None): # Added epoch
        """Called at the beginning of validation phase."""
        pass

    def on_validation_end(self, epoch: int, logs: dict[str, Any] | None = None): # Added epoch for consistency
        """Called at the end of validation phase."""
        pass

    def on_save_checkpoint(self, epoch: int, logs: dict[str, Any] | None = None):
        """Called before saving a checkpoint."""
        pass

    def on_exception(self, exception: Exception, logs: dict[str, Any] | None = None):
        """Called when an exception occurs during training."""
        pass


class MetricsLogger(Callback):
    """
    A simple callback to log metrics at the end of each epoch.
    """

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None):
        if self.engine.rank == 0 and logs:
            log_message = f"Epoch {epoch + 1} Metrics: "
            for key, value in logs.items():
                if isinstance(value, int | float):
                    log_message += f"{key}: {value:.4f} | "
                else:
                    log_message += f"{key}: {value} | "
            self.engine.logger.info(log_message.strip(" |"))


class EarlyStopping(Callback):
    """
    Callback for early stopping training if a monitored metric does not improve.

    Args:
        monitor (str): Quantity to be monitored. Default: 'val_loss'.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0.0.
        patience (int): Number of epochs with no improvement after which training will be stopped. Default: 0.
        mode (str): One of {'auto', 'min', 'max'}. In 'min' mode, training will stop when the quantity
                    monitored has stopped decreasing; in 'max' mode it will stop when the quantity
                    monitored has stopped increasing. Default: 'auto'.
        verbose (bool): Verbosity mode. Default: True.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 0,
        mode: str = "auto",
        verbose: bool = True,
    ):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = None
        self.mode = mode
        self._check_mode()

    def _check_mode(self):
        if self.mode not in ["auto", "min", "max"]:
            raise ValueError(f"EarlyStopping mode {self.mode} is unknown, 'auto', 'min' or 'max'.")
        if self.mode == "min":
            self.monitor_op = torch.lt
            self.min_delta *= -1
        elif self.mode == "max":
            self.monitor_op = torch.gt
        else:  # auto
            if "loss" in self.monitor:
                self.monitor_op = torch.lt
                self.min_delta *= -1
            else:
                self.monitor_op = torch.gt

    def on_train_start(self, logs: dict[str, Any] | None = None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = None
        if self.verbose and self.engine.rank == 0:
            self.engine.logger.info(f"EarlyStopping: Monitoring '{self.monitor}' with patience {self.patience}.")

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None):
        current = logs.get(self.monitor)
        if current is None:
            if self.verbose and self.engine.rank == 0:
                self.engine.logger.warning(
                    f"EarlyStopping: Metric '{self.monitor}' not found in logs. Skipping early stopping check."
                )
            return

        if self.best_value is None:
            self.best_value = current
        elif self.monitor_op(current - self.min_delta, self.best_value):
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.engine.should_stop_training = True
                if self.verbose and self.engine.rank == 0:
                    self.engine.logger.info(
                        f"EarlyStopping: Training stopped early at epoch {epoch + 1} due to no improvement in '{self.monitor}'."
                    )


class TensorBoardLogger(Callback):
    """
    Callback to log metrics to TensorBoard.
    """

    def __init__(self, log_dir: str = "runs"):
        super().__init__()
        self.log_dir = log_dir
        self.writer: SummaryWriter | None = None

    def on_train_start(self, logs: dict[str, Any] | None = None):
        if self.engine.rank == 0:
            log_path = Path(self.log_dir) / self.engine.config.logging.log_dir  # Use config's log_dir
            self.writer = SummaryWriter(log_path)
            self.engine.logger.info(f"TensorBoard: Logging to {log_path}")

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None):
        if self.engine.rank == 0 and self.writer and logs:
            for key, value in logs.items():
                if isinstance(value, int | float):
                    self.writer.add_scalar(f"Epoch/{key}", value, epoch)
            # Log learning rate
            if self.engine.scheduler:
                lr = self.engine.scheduler.get_last_lr()[0]
            else:
                lr = self.engine.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Epoch/LearningRate", lr, epoch)

    def on_train_step_end(
        self,
        epoch: int,
        batch_idx: int,
        loss: torch.Tensor,
        metrics: dict[str, Any],
        logs: dict[str, Any] | None = None,
    ):
        if self.engine.rank == 0 and self.writer:
            global_step = epoch * len(self.engine.dataloader) + batch_idx
            self.writer.add_scalar("Batch/Loss", loss.item(), global_step)
            for key, value in metrics.items():
                if isinstance(value, int | float):
                    self.writer.add_scalar(f"Batch/{key}", value, global_step)

    def on_train_end(self, logs: dict[str, Any] | None = None):
        if self.engine.rank == 0 and self.writer:
            self.writer.close()
            self.engine.logger.info("TensorBoard: Writer closed.")


class LRSchedulerCallback(Callback):
    """
    Callback to log learning rate at each step and epoch end.
    """

    def on_train_step_end(
        self,
        epoch: int,
        batch_idx: int,
        loss: torch.Tensor,
        metrics: dict[str, Any],
        logs: dict[str, Any] | None = None,
    ):
        if self.engine.rank == 0:
            current_lr = self.engine.optimizer.param_groups[0]["lr"]
            global_step = epoch * len(self.engine.dataloader) + batch_idx
            if (
                self.engine.config.logging.log_interval > 0
                and (batch_idx + 1) % self.engine.config.logging.log_interval == 0
            ):
                self.engine.logger.info(f"Batch {batch_idx + 1} LR: {current_lr:.6f}")

            # Log to TensorBoard if available
            if hasattr(self.engine, "callbacks"):
                for cb in self.engine.callbacks:
                    if isinstance(cb, TensorBoardLogger) and cb.writer:
                        cb.writer.add_scalar("Batch/LearningRate", current_lr, global_step)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None):
        if self.engine.rank == 0:
            current_lr = self.engine.optimizer.param_groups[0]["lr"]
            self.engine.logger.info(f"Epoch {epoch + 1} End LR: {current_lr:.6f}")

            # Log to TensorBoard if available
            if hasattr(self.engine, "callbacks"):
                for cb in self.engine.callbacks:
                    if isinstance(cb, TensorBoardLogger) and cb.writer:
                        cb.writer.add_scalar("Epoch/LearningRate_End", current_lr, epoch)
