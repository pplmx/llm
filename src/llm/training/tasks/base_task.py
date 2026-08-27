import abc
from collections.abc import Callable, Iterable
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
#
# It may return a ``logs`` dict (e.g. ``{"avg_loss": ...}``) which is
# forwarded to ``on_epoch_end`` — otherwise epoch-level observers such as
# EarlyStopping / MetricsLogger would never see the custom loop's metrics.
EpochFn = Callable[[int], dict[str, Any] | None]


class _ParameterCollection(nn.Module):
    """Expose an arbitrary parameter list as ``module.parameters()``.

    Lets :meth:`TrainingTask.build_optimizer_for` run the same
    ``build_optimizer(model)`` implementation over a parameter subset (ZeRO
    Stage-1 optimizer-state partitioning) without duplicating each task's
    group/hyperparameter logic.
    """

    def __init__(self, params: list[torch.nn.Parameter]) -> None:
        super().__init__()
        for index, param in enumerate(params):
            self.register_parameter(f"param_{index}", param)


class TrainingTask(abc.ABC, CheckpointContributor):
    """
    An abstract base class for defining a training task.

    Standard tasks implement build_* + train_step and run through TrainingEngine.
    Custom-loop tasks (e.g. PPO) set ``uses_standard_loop = False`` and override
    ``run_training``.
    """

    uses_standard_loop: bool = True

    def build_optimizer_for(self, params: Iterable[torch.nn.Parameter]) -> optim.Optimizer:
        """Build an optimizer over an arbitrary parameter subset (ZeRO Stage-1).

        ZeRO partitions optimizer state across data-parallel ranks, so the
        engine needs an optimizer built over ONLY the current rank's owned
        parameters while preserving the task's exact optimizer hyperparameters.
        The default routes ``params`` through the same :meth:`build_optimizer`
        implementation by presenting them as a minimal module, so subclasses
        get an equivalent (per-parameter) optimizer without re-implementing
        their group/hyperparameter logic. Subclasses that build optimizers from
        the model object itself (not ``model.parameters()``) should override.
        """
        return self.build_optimizer(_ParameterCollection(list(params)))

    def __init__(self, config: Config, data_module: BaseDataModule):
        self.config = config
        self.data_module = data_module

    def uses_standard_training_loop(self) -> bool:
        return self.uses_standard_loop

    def supports_pipeline_parallel(self) -> bool:
        """Whether this task's standard-loop loss matches the pipeline contract.

        Pipeline parallelism (RIL DEC-049 / TASK-210) drives the model through
        a stage schedule whose loss is exactly ``model(input_ids)`` + LM-shift
        + cross-entropy, computed on the last stage. Tasks whose
        ``train_step`` deviates — SFT passes an ``attention_mask`` into the
        model (which the pipeline's no-mask stage forward would silently
        drop), regression is not next-token loss — must opt out so the
        engine refuses PP loudly instead of training on the wrong loss.
        """
        return False

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

    def on_checkpoint_loaded(self, model: nn.Module) -> None:
        """React to the checkpointed model weights after a resume is applied.

        ``load_checkpoint`` only writes the *policy*; tasks that clone a
        frozen companion model at build time (e.g. DPO's reference) must be
        told the loaded weights here so they can re-align it — otherwise the
        companion keeps whatever random init ``build_model`` produced and the
        training signal is silently computed against noise (RIL round-60
        deep-dive Finding 1). Default: no-op.
        """

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
            # Honours the hard cap on total optimizer steps across epochs
            # for custom loops: if the previous epoch already consumed
            # ``max_steps`` steps (a smoke config), don't start another.
            max_steps = engine.config.training.max_steps
            if max_steps > 0 and engine.global_step >= max_steps:
                if engine.rank == 0:
                    engine.logger.info(f"Reached max_steps={max_steps}; stopping training at epoch {epoch}.")
                break
            engine._run_callbacks("on_epoch_start", epoch=epoch)
            epoch_logs = epoch_fn(epoch) or {}
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
