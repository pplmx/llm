import abc
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

from llm.data.base import BaseDataModule
from llm.runtime.checkpoint import CheckpointContributor
from llm.training.core.config import Config

if TYPE_CHECKING:
    from llm.training.core.callbacks import Callback
    from llm.training.core.engine import TrainingEngine

# A custom-loop epoch callback. Receives the current epoch index and runs
# the per-epoch work (data loading, optimizer steps, etc.). It is expected
# to call ``self._emit_step_callbacks(...)`` after each optimizer step so
# step-level observers (TensorBoardLogger, LRSchedulerCallback, ...) fire.
EpochFn = Callable[[int], None]


class TrainingTask(abc.ABC, CheckpointContributor):
    """
    An abstract base class for defining a training task.

    Standard tasks implement build_* + train_step and run through TrainingEngine.
    Custom-loop tasks (e.g. PPO) set ``uses_standard_loop = False`` and override
    ``run_training``.
    """

    uses_standard_loop: bool = True

    def __init__(self, config: Config, data_module: BaseDataModule):
        self.config = config
        self.data_module = data_module

    def uses_standard_training_loop(self) -> bool:
        return self.uses_standard_loop

    def prepare_training(self, engine: TrainingEngine) -> None:
        """Hook for custom-loop tasks after the model is on device."""

    def get_resume_optimizer(self) -> optim.Optimizer | None:
        """Return the optimizer used when resuming a custom training loop."""
        return None

    def build_callbacks(self) -> list[Callback]:
        """Return callbacks that should be registered on the engine.

        Subclasses override this to attach task-specific observers
        (e.g. the AdaLoRA pruning callback). The engine merges the
        returned list with whatever callbacks were passed at
        construction time and calls :meth:`Callback.set_engine` on
        each one before training starts.

        Default: no task-specific callbacks. Existing tasks are
        unaffected by this hook — it is purely additive.
        """
        return []

    def get_checkpoint_state(self) -> dict[str, Any] | None:
        return None

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        pass

    def run_training(self, engine: TrainingEngine) -> None:
        """Execute a non-standard training loop.

        Subclasses that implement a custom loop should delegate the
        per-epoch structure to :meth:`run_with_callbacks` so that
        ``on_epoch_start`` / ``on_epoch_end`` / ``on_train_step_end``
        / ``should_stop_training`` behave identically to the standard
        loop. See :class:`llm.training.tasks.ppo_task.PPOTask` for the
        canonical example.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement run_training().")

    def run_with_callbacks(self, engine: TrainingEngine, epoch_fn: EpochFn) -> None:
        """Execute a custom training loop with full callback integration.

        This wraps the per-epoch work in ``epoch_fn(epoch)`` with the same
        callback contract as :class:`TrainingEngine`'s standard loop:

        - ``on_epoch_start(epoch)`` fires before the epoch
        - ``on_epoch_end(epoch, logs={"avg_loss": ...})`` fires after
        - ``engine.should_stop_training`` is honored between epochs (set
          by callbacks like ``EarlyStopping``)
        - Exceptions propagate after ``on_exception`` fires (handled by
          ``engine.run()``)

        Inside ``epoch_fn``, call :meth:`_emit_step_callbacks` after each
        optimizer step so step-level observers fire.

        Args:
            engine: The active :class:`TrainingEngine`.
            epoch_fn: Callable invoked once per epoch with the epoch index.
        """
        epoch_logs: dict[str, Any] = {}
        for epoch in range(engine.start_epoch, engine.config.training.epochs):
            engine._run_callbacks("on_epoch_start", epoch=epoch)
            epoch_fn(epoch)
            engine._run_callbacks("on_epoch_end", epoch=epoch, logs=epoch_logs)
            epoch_logs = {}
            if engine.should_stop_training:
                if engine.rank == 0:
                    engine.logger.info(f"Training stopped early at epoch {epoch + 1} by callback.")
                break

    def _emit_step_callbacks(
        self,
        engine: TrainingEngine,
        epoch: int,
        batch_idx: int,
        loss: torch.Tensor,
        metrics: dict[str, Any],
    ) -> None:
        """Fire ``on_train_step_end`` for custom-loop tasks.

        Custom-loop tasks (PPO/RLHF) call this from inside their per-step
        code so that observers like :class:`TensorBoardLogger`,
        :class:`LRSchedulerCallback`, and :class:`EvaluationCallback`
        receive the same hook as standard tasks.

        Args:
            engine: The active :class:`TrainingEngine`.
            epoch: Current epoch index.
            batch_idx: Current batch index within the epoch.
            loss: Loss tensor from this step (synthetic tensors are OK
                for tasks that don't have a single loss, e.g. RLHF).
            metrics: Metrics dict from this step.
        """
        engine._run_callbacks(
            "on_train_step_end",
            epoch=epoch,
            batch_idx=batch_idx,
            loss=loss,
            metrics=metrics,
        )

    @abc.abstractmethod
    def build_model(self) -> nn.Module:
        """Builds and returns the model for the task."""
        pass

    @abc.abstractmethod
    def build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Builds and returns the optimizer."""
        pass

    @abc.abstractmethod
    def build_scheduler(self, optimizer: optim.Optimizer) -> LRScheduler | None:
        """Builds and returns the learning rate scheduler."""
        pass

    @abc.abstractmethod
    def build_criterion(self) -> nn.Module:
        """Builds and returns the loss function."""
        pass

    @abc.abstractmethod
    def train_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        """
        Performs a single training step.

        Returns:
            A tuple of (loss tensor, metrics dict).
        """
        pass

    @abc.abstractmethod
    def validation_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        """Performs a single validation step."""
        pass
