import operator
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from llm.core.adalora import AdaLoRAGradientEMA, prune_adalora
from llm.runtime.checkpoint import CheckpointContributor


# Forward declaration to avoid circular imports
class TrainingEngine:
    pass


class Callback(CheckpointContributor):
    """
    Base class for creating callbacks in the training engine.

    Callbacks are functions that the training engine will call at various points
    during training (e.g., at the start of training, at the end of an epoch, etc.).

    Subclasses that hold recoverable state (e.g. the AdaLoRA gradient-EMA
    tracker) override :meth:`get_checkpoint_state` and
    :meth:`load_checkpoint_state`. The engine folds callback state into
    the checkpoint ``extra_state`` automatically.
    """

    def __init__(self):
        self.engine: TrainingEngine | None = None

    def get_checkpoint_state(self) -> dict[str, Any] | None:
        """Return state to merge into the checkpoint ``extra_state``.

        Default: nothing. Override in subclasses that hold recoverable
        state (AdaLoRA EMA tracker, curriculum schedules, ...).
        """
        return None

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        """Restore state from a checkpoint ``extra_state`` fragment.

        Default: no-op. The same fragment is shared across all
        checkpoint contributors; subclasses should look up their own
        namespaced key rather than consuming the whole dict.
        """
        return None

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

    def on_optimizer_step(self, epoch: int, batch_idx: int, logs: dict[str, Any] | None = None):
        """Called after ``optimizer.step()`` but **before** ``optimizer.zero_grad()``.

        Use this hook to read parameter gradients that the next
        ``zero_grad`` call would clear — e.g. AdaLoRA's gradient-EMA
        tracker. The default is a no-op so existing callbacks are
        unaffected.
        """
        pass

    def on_validation_start(self, epoch: int, logs: dict[str, Any] | None = None):
        """Called at the beginning of validation phase."""
        pass

    def on_validation_end(self, epoch: int, logs: dict[str, Any] | None = None):
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
            self.monitor_op = operator.lt
            self.min_delta *= -1
        elif self.mode == "max":
            self.monitor_op = operator.gt
        else:  # auto
            if "loss" in self.monitor:
                self.monitor_op = operator.lt
                self.min_delta *= -1
            else:
                self.monitor_op = operator.gt

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


class EvaluationCallback(Callback):
    """Callback for periodic evaluation during training."""

    def __init__(self, eval_runner, eval_interval: int = 1000):
        super().__init__()
        self.eval_runner = eval_runner
        self.eval_interval = eval_interval

    def on_train_step_end(
        self,
        epoch: int,
        batch_idx: int,
        loss: torch.Tensor,
        metrics: dict[str, Any],
        logs: dict[str, Any] | None = None,
    ):
        """Run evaluation every N steps."""
        if (batch_idx + 1) % self.eval_interval == 0:
            results = self.eval_runner.run(self.engine.model)
            self.engine.log_metrics(results)


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


class AdaLoRAPruningCallback(Callback):
    """Adaptive-budget pruning for AdaLoRA during SFT/DPO training.

    Implements the trainer-side half of AdaLoRA Algorithm 1
    (Zhang et al. 2023, page 4). On every optimizer step the callback:

    1. Folds ``|∂L/∂λ_i|`` from each ``AdaLoRALinear`` into the EMA
       tracker (so ``update`` runs even between prune calls — pruning
       without an EMA would degenerate to magnitude-only ranking).
    2. Every ``adalora_prune_every`` steps, calls
       :func:`prune_adalora` with the linear budget schedule
       ``(adalora_tinit, adalora_tfinal)`` and the tracker's
       ``gradient_emas`` so importance scoring uses the EMA signal.
    3. Logs ``adalora/effective_rank`` (mean over AdaLoRA layers) at
       rank 0 each time the prune fires.

    The callback is a strict no-op when ``use_adalora=False``, so it is
    safe to register on every engine without an explicit feature flag.

    Args:
        use_adalora: Master switch. When ``False`` the callback does
            nothing and ``get_checkpoint_state`` returns ``None``.
        adalora_init_rank: ``init_rank`` passed to ``apply_adalora``
            by the task (kept here for clarity only).
        adalora_target_rank: ``target_rank`` — the budget the schedule
            collapses to at ``adalora_tfinal``.
        adalora_ema_alpha: EMA smoothing factor for the tracker.
        adalora_tinit: First optimizer step eligible for pruning.
        adalora_tfinal: Optimizer step at which the budget reaches
            ``adalora_target_rank``. ``None`` defers to the engine's
            step count at prune time (we just keep the budget at
            ``adalora_init_rank`` until ``tinit`` and at
            ``adalora_target_rank`` after a reasonable mid-point — see
            ``on_train_step_end`` for the exact fallback).
        adalora_prune_every: Optimizer-step cadence for the prune
            call. The EMA still updates every step.
    """

    def __init__(
        self,
        use_adalora: bool = False,
        adalora_init_rank: int = 12,
        adalora_target_rank: int = 6,
        adalora_ema_alpha: float = 0.95,
        adalora_tinit: int = 0,
        adalora_tfinal: int | None = None,
        adalora_prune_every: int = 50,
    ):
        super().__init__()
        self.use_adalora = use_adalora
        self.adalora_init_rank = adalora_init_rank
        self.adalora_target_rank = adalora_target_rank
        self.adalora_ema_alpha = adalora_ema_alpha
        self.adalora_tinit = adalora_tinit
        self.adalora_tfinal = adalora_tfinal
        self.adalora_prune_every = adalora_prune_every
        # Built in on_train_start; None while disabled.
        self._tracker: AdaLoRAGradientEMA | None = None

    def on_train_start(self, logs: dict[str, Any] | None = None):
        if not self.use_adalora:
            return
        if self.engine is None:
            return
        # Walk every AdaLoRALinear reachable from engine.model. The
        # DDP/FSDP unwrap path is ``model.modules()`` so we don't have
        # to special-case distributed wrappers.
        self._tracker = AdaLoRAGradientEMA(
            self.engine.model,
            alpha=self.adalora_ema_alpha,
        )

    def on_optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        logs: dict[str, Any] | None = None,
    ):
        """Fold gradients into the EMA **before** ``zero_grad`` clears them.

        We hook ``on_optimizer_step`` rather than ``on_train_step_end``
        because the engine calls ``optimizer.zero_grad(set_to_none=True)``
        between the two — by the time ``on_train_step_end`` fires, every
        ``.grad`` is None and the EMA would never accumulate anything.
        """
        if not self.use_adalora or self._tracker is None:
            return
        self._tracker.update()

    def on_train_step_end(
        self,
        epoch: int,
        batch_idx: int,
        loss: torch.Tensor,
        metrics: dict[str, Any],
        logs: dict[str, Any] | None = None,
    ):
        if not self.use_adalora or self._tracker is None or self.engine is None:
            return

        # Prune only on cadence. We use ``engine.global_step`` (the
        # optimizer-step counter), not the batch index, so gradient
        # accumulation doesn't bias the cadence. The EMA was already
        # updated in ``on_optimizer_step`` (which fires earlier in the
        # engine's per-batch sequence).
        global_step = getattr(self.engine, "global_step", 0)
        if global_step <= 0 or global_step % self.adalora_prune_every != 0:
            return

        # Resolve tfinal. When None, defer to ``epochs * steps_per_epoch // 2``
        # (matches the ticket spec). When ``steps_per_epoch`` is unavailable
        # we can't compute a meaningful mid-point — fall back to collapsing
        # the budget to ``adalora_target_rank`` immediately on the next prune.
        # We never recompute tfinal per-call (a moving tfinal means the
        # budget never converges).
        tfinal = self.adalora_tfinal
        if tfinal is None:
            cfg = getattr(self.engine, "config", None)
            try:
                steps_per_epoch = int(
                    getattr(getattr(cfg, "data", None), "steps_per_epoch", 0) or 0  # type: ignore[union-attr]
                )
                epochs = int(getattr(cfg.training, "epochs", 1) or 1)  # type: ignore[union-attr]
            except (AttributeError, TypeError, ValueError):
                steps_per_epoch = 0
                epochs = 1
            if steps_per_epoch > 0:
                tfinal = max(self.adalora_tinit + 1, (epochs * steps_per_epoch) // 2)
            else:
                # No schedule — collapse straight to target_rank so the
                # user gets *some* pruning on cadence. They can set
                # ``adalora_tfinal`` explicitly for a proper schedule.
                tfinal = global_step

        prune_adalora(
            self.engine.model,
            schedule=(self.adalora_tinit, tfinal),
            current_step=global_step,
            gradient_emas=self._tracker.as_dict(),
        )

        # Log mean effective rank so the trainer can see pruning happen.
        if self.engine.rank == 0:
            from llm.core.adalora import AdaLoRALinear

            ranks = [
                m.effective_rank
                for m in self.engine.model.modules()
                if isinstance(m, AdaLoRALinear)
            ]
            if ranks:
                mean_rank = sum(ranks) / len(ranks)
                self.engine.logger.info(f"adalora/effective_rank={mean_rank:.1f}")

    def get_checkpoint_state(self) -> dict[str, Any] | None:
        if self._tracker is None:
            return None
        return self._tracker.state_dict()

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        if self._tracker is None:
            return
        self._tracker.load_state_dict(state)


class PEFTAdapterCheckpointCallback(Callback):
    """Auto-save PEFT adapter weights to a sidecar file at training end.

    Built on top of the T2 PEFT #47 :func:`save_peft` / :func:`load_peft`
    helpers. At :meth:`on_train_end`, writes the model's trainable
    adapter parameters to ``peft_save_path`` using the same envelope
    format (:data:`~llm.core.peft.checkpoint.PEFT_CHECKPOINT_FORMAT_VERSION`),
    so a fresh model can recover the trained adapter via
    :func:`load_peft` without going through the full
    :class:`CheckpointManager` flow.

    The sidecar is separate from the main checkpoint because:

    1. **Adapter weights are tiny compared to base weights** — saving
       them as a sidecar enables "share just the adapter" workflows
       (cross-base-model transfer, adapter-only inference).
    2. **The main checkpoint already preserves adapter weights** inside
       ``model_state`` for standard resume — no duplication needed.

    The callback is a strict no-op when ``peft_method`` is ``None`` or
    ``peft_save_path`` is ``None``, so it is safe to register
    conditionally (the task only wires it when ``peft_method`` is
    actually set on the config).

    Loading is **NOT** automatic on resume — the standard
    ``CheckpointManager.load_checkpoint`` already restores the full
    ``model_state`` (which contains the adapter weights). Users who
    want cross-base-model adapter transfer (no main checkpoint, just
    the sidecar) should call :func:`load_peft` explicitly after
    :meth:`LanguageModelingTask.build_model`.

    Args:
        peft_method: Registered PEFT method name (e.g. ``"lora"``).
            When ``None``, the callback is a strict no-op.
        peft_kwargs: Method-specific kwargs forwarded verbatim to
            :func:`save_peft` — e.g. ``{"rank": 8, "alpha": 16.0}``
            for LoRA. Stored in the sidecar envelope so a future
            :func:`load_peft` knows how to re-apply the method.
        peft_save_path: Where to write the adapter file. When
            ``None``, :meth:`on_train_end` does nothing (config
            opt-out).
    """

    def __init__(
        self,
        peft_method: str | None = None,
        peft_kwargs: dict[str, Any] | None = None,
        peft_save_path: str | Path | None = None,
    ):
        super().__init__()
        self.peft_method = peft_method
        # Defensive copy + dict() cast — caller may pass a Pydantic
        # model attribute that we shouldn't mutate.
        self.peft_kwargs: dict[str, Any] = dict(peft_kwargs or {})
        self.peft_save_path = (
            Path(peft_save_path) if peft_save_path is not None else None
        )

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """Write the adapter sidecar. No-op when not configured.

        Failure is logged + swallowed rather than re-raised — the main
        checkpoint has already been written by the time
        ``on_train_end`` fires, and losing the sidecar is recoverable
        (re-train, or use the main checkpoint). Crashing here would
        lose the main checkpoint too.
        """
        if self.peft_method is None or self.peft_save_path is None:
            return
        if self.engine is None:
            return
        try:
            # Lazy import — the callback module is on the trainer hot
            # path; we don't want to import the PEFT stack (and torch's
            # PEFT helper chain) until we actually need to save.
            from llm.core.peft.checkpoint import save_peft

            save_peft(
                self.engine.model,
                self.peft_save_path,
                self.peft_method,
                **self.peft_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 — see docstring
            # ``logger.warning`` may not exist on a stub engine in
            # tests, so guard the call.
            logger = getattr(self.engine, "logger", None)
            if logger is not None:
                logger.warning(
                    f"PEFT adapter sidecar save failed: {exc}. "
                    f"Main checkpoint is unaffected. Path={self.peft_save_path}"
                )

    def get_checkpoint_state(self) -> dict[str, Any] | None:
        """Stash the sidecar path in the main checkpoint's ``extra_state``.

        The actual adapter weights live on disk in the sidecar file
        itself — we only need to remember WHERE the sidecar is so a
        resumed run knows where to write the next one.
        """
        if self.peft_save_path is None:
            return None
        return {"peft_save_path": str(self.peft_save_path)}

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        """Restore the sidecar path from a resumed checkpoint."""
        if state is None:
            return
        path_str = state.get("peft_save_path") if isinstance(state, dict) else None
        if path_str is not None:
            self.peft_save_path = Path(path_str)
