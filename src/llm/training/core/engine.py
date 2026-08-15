import sys
import time
from typing import Any, cast

import torch
import torch.nn as nn

from llm.data.base import BaseDataModule
from llm.runtime.checkpoint import collect_extra_state, load_extra_state
from llm.training.core.callbacks import Callback
from llm.training.core.config import Config
from llm.training.core.distributed import broadcast_parameters
from llm.training.core.utils import CheckpointManager, DistributedManager, Logger, PerformanceMonitor
from llm.training.distributed import model_for_checkpoint_io, wrap_model_for_training
from llm.training.tasks.base_task import TrainingTask
from llm.utils.common import count_parameters

_MIN_FREE_VRAM_BYTES = 512 * 1024 * 1024  # 512 MiB headroom for CUDA context + model


def _cuda_usable(device_idx: int = 0) -> bool:
    """True only if a CUDA device is available *and* has allocatable VRAM.

    ``torch.cuda.is_available()`` can return True in containers that report
    CUDA devices but have 0 usable VRAM (CUDA OOM on first allocation).  Even
    a non-zero free-byte count can be misleading: the CUDA driver reserves
    context memory, so a device reporting only a few hundred MiB free will OOM
    on the first real tensor allocation.  This helper rejects such devices by
    requiring at least ``_MIN_FREE_VRAM_BYTES`` of free VRAM on the specific
    device index that the caller intends to use.
    """
    if not torch.cuda.is_available():
        return False
    if device_idx >= torch.cuda.device_count():
        return False
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device_idx)
    except RuntimeError, torch.AcceleratorError:
        return False
    return free_bytes >= _MIN_FREE_VRAM_BYTES


class TrainingEngine:
    def __init__(
        self,
        config: Config,
        task: TrainingTask,
        rank: int,
        world_size: int,
        data_module: BaseDataModule,
        callbacks: list[Callback] | None = None,
    ):
        self.config = config
        self.task = task
        self.rank = rank
        self.world_size = world_size

        if torch.cuda.is_available() and torch.cuda.device_count() > 0 and self.world_size > 0:
            cuda_idx = rank % torch.cuda.device_count()
            # world_size > 0 is a proxy for intending to use GPUs if available
            if _cuda_usable(cuda_idx):
                self.device = torch.device(f"cuda:{cuda_idx}")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.data_module = data_module

        self.logger = Logger(rank, config.logging)
        self.performance_monitor = PerformanceMonitor(rank, self.device)
        self.checkpoint_manager = CheckpointManager(config.checkpoint, rank, self.logger)
        self.callbacks = callbacks or []
        for callback in self.callbacks:
            callback.set_engine(cast(TrainingEngine, self))

        self._setup_components()

    def _register_task_callbacks(self) -> None:
        """Append and wire any task-supplied callbacks.

        Called after ``_setup_components`` so :meth:`build_model` has
        already applied feature gates (e.g. AdaLoRA layers) that the
        callbacks may need to introspect. We append *after* the
        constructor-passed callbacks so task callbacks run last in the
        ordered callback list.
        """
        for callback in self.task.build_callbacks():
            callback.set_engine(cast(TrainingEngine, self))
            self.callbacks.append(callback)
        self.training_start_time = time.time()
        self.should_stop_training = False
        self.global_step = 0

    def get_checkpoint_state(self) -> dict[str, Any] | None:
        """Persist the global optimizer-step counter across a resume.

        Without this, every resume restarts ``global_step`` at 0: a
        ``max_steps`` budget re-arms and trains a fresh full run on top of
        the completed steps, the AdaLoRA prune cadence restarts (and one
        round later can crash a resumed run when the reduced-effective-rank
        layers can't un-prune back up to ``init_rank``), and
        TensorBoard/LRScheduler x-indices duplicate across the resume
        boundary (RIL round-47 deep-dive).
        """
        return {"engine.global_step": self.global_step}

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        """Restore ``global_step`` from a checkpoint's merged extra_state.

        Called by ``load_extra_state`` during ``_setup_components`` — after
        ``_register_task_callbacks`` resets the counter to 0 — so the
        restored value wins and ``max_steps`` / AdaLoRA / logging cadences
        resume where they left off.
        """
        if not state:
            return
        restored = state.get("engine.global_step")
        if restored is not None:
            self.global_step = int(restored)

    def _run_callbacks(self, method_name: str, *args, **kwargs):
        for callback in self.callbacks:
            getattr(callback, method_name)(*args, **kwargs)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Log an evaluation/metrics dict at rank 0.

        Used by :class:`EvaluationCallback` to surface periodic evaluation
        results. Numeric values are formatted like the batch-stats log;
        non-numeric values (e.g. ``num_samples``) are included as-is.
        """
        if self.rank != 0 or not metrics:
            return
        parts: list[str] = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                parts.append(f"{key}: {value:.4f}")
            else:
                parts.append(f"{key}: {value}")
        self.logger.info("Evaluation metrics: " + " | ".join(parts))

    def _setup_components(self):
        """Builds all necessary components for training from the task."""
        self.logger.info("Setting up training components...")

        # Build model and move to device
        model = self.task.build_model().to(self.device)
        if self.rank == 0:
            total, trainable = count_parameters(model)  # Use the utility function
            self.logger.info(f"🏗️  Model: {total:,} total params, {trainable:,} trainable")

        # DDP requires every rank to start from *identical* parameters.  Our
        # DistributedManager.setup seeds rank R's RNG with ``42 + R`` (so data
        # ordering differs per rank), which means a freshly built model also
        # RNG-initialises differently on each rank — and DDP only averages
        # *gradients*, it does not reconcile divergent weights.  Without an
        # explicit broadcast, a fresh multi-GPU run optimises a different
        # model per rank and never converges on the real loss.  Broadcast
        # rank 0's parameters now, before the model is wrapped or compiled.
        # (Resumed runs are already in sync via the shared checkpoint; this
        # broadcast is a cheap no-op for them.)
        broadcast_parameters(model)

        # Tasks that clone a frozen reference model inside ``build_model``
        # (e.g. :class:`~llm.training.tasks.dpo_task.DPOTask`) copy the
        # policy *before* the broadcast above, so their reference carries
        # each rank's own RNG-initialised weights. Sync it from rank 0 too,
        # otherwise multi-GPU DPO computes per-rank-divergent reference
        # logps and the loss is mathematically wrong (RIL ISS-038).
        # Tasks that build their reference later (PPO deep-copies the
        # already-broadcast policy) expose no ``ref_model`` here and are
        # unaffected.
        ref_model = getattr(self.task, "ref_model", None)
        if ref_model is not None:
            broadcast_parameters(ref_model)

        # Compile model if enabled. Mode and dynamic-shape marking come from
        # OptimizationConfig (Finding AL). Default mode is "default" — the
        # previous hardcoded "reduce-overhead" used CUDA graphs, which is
        # incompatible with variable-length sequences and KV-cache eviction.
        # Users who want CUDA-graph capture can still opt in explicitly.
        if self.config.optimization.use_compile and sys.version_info >= (3, 8):
            self.logger.info(
                f"🚀 Compiling model with torch.compile (mode={self.config.optimization.compile_mode!r})..."
            )
            compile_kwargs: dict[str, Any] = {"mode": self.config.optimization.compile_mode}
            if self.config.optimization.compile_dynamic is not None:
                compile_kwargs["dynamic"] = self.config.optimization.compile_dynamic
            try:
                model = torch.compile(model, **compile_kwargs)
            except (RuntimeError, TypeError, AttributeError) as e:
                self.logger.warning(f"torch.compile failed: {e}. Continuing without it.")

        # ``torch.compile`` may reassign ``model`` and confuse the type
        # checker about its concrete type; narrow it back before wrapping.
        if not isinstance(model, nn.Module):
            raise TypeError(f"build_model() must return an nn.Module, got {type(model).__name__}")
        self.model = wrap_model_for_training(
            model,
            parallel_strategy=self.config.distributed.parallel_strategy,
            device=self.device,
            world_size=self.world_size,
        )

        self.use_standard_loop = self.task.uses_standard_training_loop()

        # Use data_module to get dataloaders
        self.is_streaming = getattr(self.data_module, "is_streaming", False)
        self.dataloader, self.sampler = self.data_module.train_dataloader(self.rank, self.world_size)
        self.val_dataloader, self.val_sampler = self.data_module.val_dataloader(self.rank, self.world_size)

        if self.is_streaming and self.config.data.steps_per_epoch is None:
            raise ValueError("Streaming DataModules require data.steps_per_epoch to be set.")

        if self.use_standard_loop:
            self.optimizer = self.task.build_optimizer(self.model)
            self.scheduler = self.task.build_scheduler(self.optimizer)
            self.criterion = self.task.build_criterion().to(self.device)
        else:
            self.optimizer = None
            self.scheduler = None
            self.criterion = None

        # Narrow the possibly-None standard-loop attributes: the standard
        # loop guarantees all three are set.
        if self.use_standard_loop and (self.optimizer is None or self.scheduler is None or self.criterion is None):
            raise RuntimeError("standard-loop components were not built")

        # Resolve 'auto' dtype
        self.resolved_amp_dtype = self.config.optimization.amp_dtype
        if self.resolved_amp_dtype == "auto":
            if self.device.type == "cuda":
                if torch.cuda.is_bf16_supported():
                    self.resolved_amp_dtype = "bfloat16"
                    self.logger.info("✨ Auto-detected BF16 support. Using bfloat16 for AMP.")
                else:
                    self.resolved_amp_dtype = "float16"
                    self.logger.info("✨ Auto-detected no BF16 support. Using float16 for AMP.")
            else:
                self.resolved_amp_dtype = "float32"
                self.logger.info("✨ Using float32 (no AMP) on CPU.")

        # BF16 typically doesn't need scaling
        use_scaler = (
            self.config.optimization.use_amp and self.device.type == "cuda" and self.resolved_amp_dtype == "float16"
        )
        self.scaler = torch.amp.GradScaler(enabled=use_scaler)

        # Task-supplied callbacks (e.g. AdaLoRA pruning) are wired last
        # so they see the wrapped, device-resident model.
        self._register_task_callbacks()

        model_to_load = model_for_checkpoint_io(self.model)

        if self.use_standard_loop:
            self.start_epoch, self.best_loss = self.checkpoint_manager.load_checkpoint(
                model_to_load, self.optimizer, self.scheduler, self.scaler, self.device
            )
            load_extra_state(
                self.checkpoint_manager.loaded_extra_state,
                self,
                self.data_module,
                self.task,
                *self.callbacks,
            )
        else:
            self.start_epoch = 0
            self.best_loss = float("inf")
            self.task.prepare_training(self)
            if self.config.checkpoint.resume_from_checkpoint:
                resume_optimizer = self.task.get_resume_optimizer()
                if resume_optimizer is None:
                    raise ValueError(
                        f"{type(self.task).__name__} must implement get_resume_optimizer() when resuming training."
                    )
                self.start_epoch, self.best_loss = self.checkpoint_manager.load_checkpoint(
                    model_to_load,
                    resume_optimizer,
                    None,
                    self.scaler,
                    self.device,
                )
                load_extra_state(
                    self.checkpoint_manager.loaded_extra_state,
                    self,
                    self.data_module,
                    self.task,
                    *self.callbacks,
                )

        # ``load_checkpoint`` only ever writes the POLICY (model_to_load).
        # Tasks that clone a frozen companion at build time — DPO's reference
        # model snapshots the *initial* policy in ``build_model``, BEFORE the
        # checkpoint is applied — must re-align that companion with the loaded
        # weights here, or a resumed run silently computes its training signal
        # against a random model (RIL round-60 deep-dive Finding 1). Runs with
        # no resume apply a fresh random policy, so the task's own build-time
        # snapshot is already correct and the hook is a no-op for them.
        self.task.on_checkpoint_loaded(model_to_load)

        self.checkpoint_manager.best_loss = self.best_loss

    def _iter_training_batches(self):
        if self.is_streaming:
            data_iter = iter(self.dataloader)
            num_batches = self.config.data.steps_per_epoch
            if num_batches is None:
                raise RuntimeError("streaming requires data.steps_per_epoch")
            for batch_idx in range(num_batches):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # Corpus exhausted before the step budget: streaming
                    # pretraining cycles the corpus, so restart the source
                    # from the beginning and keep going.
                    dataset = self.dataloader.dataset
                    reset = getattr(dataset, "reset", None)
                    if reset is None:
                        raise RuntimeError(
                            "streaming dataloader exhausted before steps_per_epoch "
                            f"and its dataset ({type(dataset).__name__}) has no reset()"
                        ) from None
                    reset()
                    data_iter = iter(self.dataloader)
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        raise RuntimeError("streaming corpus is empty; nothing to train on") from None
                yield batch_idx, batch, num_batches
            return

        num_batches = len(self.dataloader)
        for batch_idx, batch in enumerate(self.dataloader):
            yield batch_idx, batch, num_batches

    def _run_epoch(self, epoch: int) -> float:
        if self.optimizer is None or self.criterion is None:
            raise RuntimeError("standard-loop components are required")
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        self.model.train()
        self.performance_monitor.reset_epoch_stats()

        epoch_loss = 0.0
        accum_steps = self.config.optimization.gradient_accumulation_steps
        batch_count = 0
        # Grad-accumulation window micro-batch counter. Each loss is scaled
        # by 1/accum_steps, so a FULL window accumulates one global-batch's
        # average gradient at step time. When an epoch's tail is a partial
        # window (num_batches % accum_steps != 0) the forced final step
        # would otherwise carry only window_count/accum_steps of the intended
        # magnitude — a silent, step-to-step-varying effective-LR reduction
        # (RIL ISS-066). We track the count and re-scale at step time.
        accum_counter = 0

        # Zero gradients at start of epoch
        self.optimizer.zero_grad(set_to_none=True)

        max_steps = self.config.training.max_steps
        for batch_idx, batch, num_batches in self._iter_training_batches():
            if max_steps > 0 and self.global_step >= max_steps:
                if self.rank == 0:
                    self.logger.info(f"Reached max_steps={max_steps}; stopping training.")
                break
            # Count only batches actually processed. ``num_batches`` is the
            # full-epoch length, so when ``max_steps`` truncates the last
            # epoch mid-way the average must divide by the processed count —
            # otherwise the partial epoch's avg_loss is deflated (which
            # corrupts save_best / EarlyStopping / ReduceLROnPlateau, RIL
            # round-47 deep-dive).
            batch_count += 1
            self._run_callbacks("on_batch_start", epoch=epoch, batch_idx=batch_idx)
            batch_start_time = time.time()
            # Move batch to device
            if isinstance(batch, dict):
                batch = {
                    k: v.to(self.device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)
                }
            else:
                batch = tuple(t.to(self.device, non_blocking=True) for t in batch)

            amp_dtype = torch.float16
            if self.resolved_amp_dtype == "bfloat16":
                amp_dtype = torch.bfloat16

            with torch.autocast(
                device_type=self.device.type, enabled=self.config.optimization.use_amp, dtype=amp_dtype
            ):
                loss, metrics = self.task.train_step(batch, self.model, self.criterion)
                # Scale loss for gradient accumulation
                loss = loss / accum_steps

            self.scaler.scale(loss).backward()
            accum_counter += 1

            # Perform optimization step every accum_steps or at end of epoch
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1 == num_batches):
                self.scaler.unscale_(self.optimizer)
                if accum_counter != accum_steps:
                    # Partial final window (epoch tail): every loss was scaled
                    # by 1/accum_steps, so this step only accumulated
                    # `accum_counter/accum_steps` of a global batch. Re-scale
                    # the gradient to full-batch-average magnitude so the
                    # effective learning rate does not silently drop on
                    # partial windows (RIL ISS-066).
                    _scale = accum_steps / accum_counter
                    for _param in self.model.parameters():
                        if _param.grad is not None:
                            _param.grad.mul_(_scale)
                accum_counter = 0
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip_val
                )
                # GradScaler silently skips the real parameter update when a
                # gradient is inf/NaN (found_inf, recorded during unscale_).
                # ``clip_grad_norm_`` computes its total norm from the same
                # unscaled gradients, so a non-finite ``grad_norm`` is exactly
                # the scaler's skip signal — detected here with zero extra
                # syncs (RIL ISS-107). A skipped step must NOT advance
                # ``global_step`` (that counter drives max_steps and the
                # AdaLoRA prune cadence) and must NOT fire
                # ``on_optimizer_step`` — the AdaLoRA gradient-EMA would fold
                # ``|inf/NaN|`` into its running importance and stay
                # NaN-poisoned forever (``alpha * NaN = NaN``).
                step_skipped = not bool(torch.isfinite(grad_norm))
                if step_skipped and not self.scaler.is_enabled():
                    # No GradScaler protection (fp32/bf16/CPU): refusing the
                    # update ourselves keeps poisoned gradients out of the
                    # parameters instead of writing NaN weights.
                    pass
                else:
                    self.scaler.step(self.optimizer)
                # update() must still run on the skipped path: it advances the
                # scaler's internal stage machine and decays the loss scale.
                self.scaler.update()
                if not step_skipped:
                    # Fire ``on_optimizer_step`` BEFORE zero_grad so
                    # gradient-reading observers (AdaLoRA EMA tracker) get
                    # the real gradients, not the post-zero zeros.
                    self._run_callbacks("on_optimizer_step", epoch=epoch, batch_idx=batch_idx)
                self.optimizer.zero_grad(set_to_none=True)
                if not step_skipped:
                    self.global_step += 1
            else:
                # Calculate grad norm even if not stepping? Or skip?
                # Usually we track grad norm of the step.
                # Here we just use 0.0 or skip logging grad norm for micro-steps?
                # Logic: We only log roughly.
                grad_norm = torch.tensor(0.0)

            self._run_callbacks("on_train_step_end", epoch=epoch, batch_idx=batch_idx, loss=loss, metrics=metrics)

            # NOTE: loss logged is scaled loss or raw loss?
            # Usually we want raw loss. Task returns raw loss.
            # We scaled `loss` variable.
            # metrics['loss'] is usually raw loss item.
            # If task returns loss tensor, metrics dict might satisfy raw loss.

            batch_loss = metrics.get("loss", loss.item() * accum_steps)  # Restore scale for logging if needed
            epoch_loss += batch_loss

            # Performance monitoring
            self.performance_monitor.log_batch_time(time.time() - batch_start_time)
            self.performance_monitor.log_loss(batch_loss)
            self.performance_monitor.log_gradient_norm(grad_norm.item())

            if (batch_idx + 1) % self.config.logging.log_interval == 0 and self.rank == 0:
                self._log_batch_stats(epoch, batch_idx, num_batches, metrics)
            self._run_callbacks("on_batch_end", epoch=epoch, batch_idx=batch_idx)

        if batch_count == 0:
            return 0.0

        loss_tensor = torch.tensor(epoch_loss / batch_count, device=self.device)
        global_avg_loss = DistributedManager.reduce_mean(loss_tensor).item()

        return global_avg_loss

    def _run_validation_epoch(self, epoch: int) -> float | None:
        if self.criterion is None:
            raise RuntimeError("standard-loop components are required")
        self._run_callbacks("on_validation_start", epoch=epoch)
        self.model.eval()  # Set model to evaluation mode
        self.performance_monitor.reset_epoch_stats()

        val_loss = 0.0
        # Use val_dataloader if available, otherwise skip validation
        if self.val_dataloader is None:
            self.logger.warning("Validation dataloader not provided. Skipping validation.")
            self._run_callbacks("on_validation_end", epoch=epoch, logs={"val_loss": None})
            return None

        if self.val_sampler:  # Set epoch for validation sampler if it exists
            self.val_sampler.set_epoch(epoch)

        num_batches = len(self.val_dataloader)
        if num_batches == 0:
            # An empty validation split is not an error — skip validation
            # instead of dividing by zero below.
            self.logger.warning("Validation dataloader is empty. Skipping validation.")
            self._run_callbacks("on_validation_end", epoch=epoch, logs={"val_loss": None})
            return None

        with torch.no_grad():  # Disable gradient calculations
            for batch_idx, batch in enumerate(self.val_dataloader):
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {
                        k: v.to(self.device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)
                    }
                else:
                    batch = tuple(t.to(self.device, non_blocking=True) for t in batch)

                loss, metrics = self.task.validation_step(batch, self.model, self.criterion)

                batch_loss = metrics.get("loss", loss.item())
                val_loss += batch_loss

                # Performance monitoring (optional for validation)
                self.performance_monitor.log_loss(batch_loss)

                if (batch_idx + 1) % self.config.logging.log_interval == 0 and self.rank == 0:
                    self._log_batch_stats(epoch, batch_idx, num_batches, metrics)  # Reuse log_batch_stats

        loss_tensor = torch.tensor(val_loss / num_batches, device=self.device)
        global_avg_loss = DistributedManager.reduce_mean(loss_tensor).item()
        self._run_callbacks("on_validation_end", epoch=epoch, logs={"val_loss": global_avg_loss})

        return global_avg_loss

    def _log_batch_stats(self, epoch, batch_idx, num_batches, metrics):
        if self.optimizer is None:
            raise RuntimeError("standard-loop components are required")
        lr = self.optimizer.param_groups[0]["lr"]
        mem_alloc, mem_cached = self.performance_monitor.get_current_gpu_memory()
        grad_norm = self.performance_monitor.gradient_norms[-1] if self.performance_monitor.gradient_norms else 0.0
        batch_time = self.performance_monitor.get_avg_batch_time()

        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

        log_msg = (
            f"Epoch {epoch + 1:2d} | Batch {batch_idx + 1:4d}/{num_batches} | "
            f"{metrics_str} | Grad Norm: {grad_norm:.2f} | "
            f"LR: {lr:.6f} | B-Time: {batch_time * 1000:.1f}ms | "
            f"Mem: {mem_alloc:.2f}/{mem_cached:.2f} GB"
        )
        self.logger.info(log_msg)

    def run(self):
        self._run_callbacks("on_train_start")
        if self.rank == 0:
            self.logger.info("🎉 Starting training...")

        if not self.use_standard_loop:
            try:
                self.task.run_training(self)
            except Exception as e:
                self._run_callbacks("on_exception", exception=e)
                raise
            finally:
                if self.rank == 0:
                    total_time = time.time() - self.training_start_time
                    self.logger.info(f"✅ Training completed in {total_time / 3600:.2f} hours.")
                self._run_callbacks("on_train_end")
            return

        try:
            if self.optimizer is None:
                raise RuntimeError("standard-loop components are required")
            for epoch in range(self.start_epoch, self.config.training.epochs):
                self._run_callbacks("on_epoch_start", epoch=epoch)
                epoch_start_time = time.time()
                avg_loss = self._run_epoch(epoch)
                val_loss = None
                if self.config.training.run_validation:
                    val_loss = self._run_validation_epoch(epoch)

                if self.should_stop_training:  # Check early stopping flag
                    if self.rank == 0:
                        self.logger.info(f"Training stopped early at epoch {epoch + 1} by EarlyStopping callback.")
                    break  # Break the training loop

                if self.scheduler:
                    # ReduceLROnPlateau needs the metric, others don't
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(
                            val_loss if val_loss is not None else avg_loss
                        )  # Use val_loss for scheduler if available
                    else:
                        self.scheduler.step()

                DistributedManager.barrier()

                if self.rank == 0:
                    epoch_time = time.time() - epoch_start_time
                    lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]["lr"]
                    peak_mem = self.performance_monitor.get_peak_gpu_memory()
                    self.logger.info("-" * 80)
                    log_msg = (
                        f"Epoch {epoch + 1:2d}/{self.config.training.epochs} SUMMARY | Train Loss: {avg_loss:.4f} | "
                    )
                    if val_loss is not None:
                        log_msg += f"Val Loss: {val_loss:.4f} | "
                    log_msg += f"LR: {lr:.6f} | Time: {epoch_time:.2f}s | Peak Mem: {peak_mem:.2f} GB"
                    self.logger.info(log_msg)
                    self.logger.info("-" * 80)

                # Collect extra state on ALL ranks before the rank-0 save gate:
                # streaming DataModules gather per-rank shard cursors
                # collectively (all_gather_object) so the checkpoint persisted
                # by rank 0 holds every rank's resume cursor. Calling this only
                # on rank 0 would drop all other shards (silent re-training
                # from line 0 on resume) and would deadlock any collective.
                extra_state = collect_extra_state(self, self.data_module, self.task, *self.callbacks)

                if self.rank == 0:
                    # Save checkpoint based on validation loss if available, otherwise training loss
                    metric_for_checkpoint = val_loss if val_loss is not None else avg_loss
                    self.checkpoint_manager.save_checkpoint(
                        epoch,
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.scaler,
                        metric_for_checkpoint,
                        extra_state=extra_state,
                        model_config=self.config.model.model_dump(),
                    )
                    self._run_callbacks("on_save_checkpoint", epoch=epoch)

                logs = {"avg_loss": avg_loss}
                if val_loss is not None:
                    logs["val_loss"] = val_loss
                self._run_callbacks("on_epoch_end", epoch=epoch, logs=logs)

                # Check max_steps AFTER checkpoint save + epoch-end callbacks
                # so the checkpoint for the current epoch is persisted even
                # when max_steps is hit mid-epoch.  Moving this before the
                # save would silently drop the checkpoint (verified by
                # test_dpo_runs_and_saves_checkpoint).
                if self.config.training.max_steps > 0 and self.global_step >= self.config.training.max_steps:
                    if self.rank == 0:
                        self.logger.info(f"Reached max_steps={self.config.training.max_steps}; training complete.")
                    break

        except Exception as e:
            self._run_callbacks("on_exception", exception=e)
            raise

        finally:
            if self.rank == 0:
                total_time = time.time() - self.training_start_time
                self.logger.info(f"✅ Training completed in {total_time / 3600:.2f} hours on {self.world_size} GPUs.")
                self.logger.info(f"🌟 Best loss achieved: {self.checkpoint_manager.best_loss:.4f}")
            self._run_callbacks("on_train_end")
