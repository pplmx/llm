import sys
import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from llm.data.data_module import BaseDataModule  # Added BaseDataModule
from llm.training.core.callbacks import Callback
from llm.training.core.config import Config
from llm.training.core.utils import CheckpointManager, DistributedManager, Logger, PerformanceMonitor
from llm.training.tasks.base_task import TrainingTask
from llm.utils.common import count_parameters  # Added import


class TrainingEngine:
    def __init__(
        self,
        config: Config,
        task: TrainingTask,
        rank: int,
        world_size: int,
        data_module: BaseDataModule,  # Changed to data_module
        callbacks: list[Callback] | None = None,
    ):
        self.config = config
        self.task = task
        self.rank = rank
        self.world_size = world_size

        if torch.cuda.is_available() and torch.cuda.device_count() > 0 and self.world_size > 0:
            # world_size > 0 is a proxy for intending to use GPUs if available
            # rank % torch.cuda.device_count() ensures valid device index per process
            self.device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        else:
            self.device = torch.device("cpu")

        self.data_module = data_module  # Stored data_module

        self.logger = Logger(rank, config.logging)
        self.performance_monitor = PerformanceMonitor(rank, self.device)
        self.checkpoint_manager = CheckpointManager(config.checkpoint, rank, self.logger)
        self.callbacks = callbacks or []
        for callback in self.callbacks:
            callback.set_engine(self)

        self._setup_components()
        self.training_start_time = time.time()
        self.should_stop_training = False  # Added for early stopping

    def _run_callbacks(self, method_name: str, *args, **kwargs):
        for callback in self.callbacks:
            getattr(callback, method_name)(*args, **kwargs)

    def _setup_components(self):
        """Builds all necessary components for training from the task."""
        self.logger.info("Setting up training components...")

        # Build model and move to device
        model = self.task.build_model().to(self.device)
        if self.rank == 0:
            total, trainable = count_parameters(model)  # Use the utility function
            self.logger.info(f"🏗️  Model: {total:,} total params, {trainable:,} trainable")

        # Compile model if enabled
        if self.config.optimization.use_compile and sys.version_info >= (3, 8):
            self.logger.info("🚀 Compiling model with torch.compile...")
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}. Continuing without it.")

        # DDP wrapper
        if self.world_size > 1 and self.device.type == "cuda":
            # Ensure device_ids is only passed if actually using CUDA and DDP
            self.model = DDP(model, device_ids=[self.device.index], find_unused_parameters=False)
        else:
            self.model = model  # No DDP for CPU or single GPU / single process

        # Build other components from task
        self.optimizer = self.task.build_optimizer(self.model)  # Optimizer should work with model or model.module
        self.scheduler = self.task.build_scheduler(self.optimizer)
        self.criterion = self.task.build_criterion().to(self.device)

        # Use data_module to get dataloaders
        self.dataloader, self.sampler = self.data_module.train_dataloader(self.rank, self.world_size)
        self.val_dataloader, self.val_sampler = self.data_module.val_dataloader(self.rank, self.world_size)

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

        # Determine the model to pass for checkpoint loading (unwrap if DDP)
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model

        self.start_epoch, self.best_loss = self.checkpoint_manager.load_checkpoint(
            model_to_load, self.optimizer, self.scheduler, self.scaler, self.device
        )
        self.checkpoint_manager.best_loss = self.best_loss

    def _run_epoch(self, epoch: int) -> float:
        self.sampler.set_epoch(epoch)
        self.model.train()
        self.performance_monitor.reset_epoch_stats()

        epoch_loss = 0.0
        num_batches = len(self.dataloader)

        for batch_idx, batch in enumerate(self.dataloader):
            self._run_callbacks("on_batch_start", epoch=epoch, batch_idx=batch_idx)
            batch_start_time = time.time()
            # Move batch to device
            batch = tuple(t.to(self.device, non_blocking=True) for t in batch)

            self.optimizer.zero_grad(set_to_none=True)

            amp_dtype = torch.float16
            if self.resolved_amp_dtype == "bfloat16":
                amp_dtype = torch.bfloat16

            with torch.autocast(
                device_type=self.device.type, enabled=self.config.optimization.use_amp, dtype=amp_dtype
            ):
                loss, metrics = self.task.train_step(batch, self.model, self.criterion)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip_val)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._run_callbacks("on_train_step_end", epoch=epoch, batch_idx=batch_idx, loss=loss, metrics=metrics)

            batch_loss = metrics.get("loss", loss.item())
            epoch_loss += batch_loss

            # Performance monitoring
            self.performance_monitor.log_batch_time(time.time() - batch_start_time)
            self.performance_monitor.log_loss(batch_loss)
            self.performance_monitor.log_gradient_norm(grad_norm.item())

            if (batch_idx + 1) % self.config.logging.log_interval == 0 and self.rank == 0:
                self._log_batch_stats(epoch, batch_idx, num_batches, metrics)
            self._run_callbacks("on_batch_end", epoch=epoch, batch_idx=batch_idx)

        loss_tensor = torch.tensor(epoch_loss / num_batches, device=self.device)
        global_avg_loss = DistributedManager.reduce_mean(loss_tensor).item()

        return global_avg_loss

    def _run_validation_epoch(self, epoch: int) -> float | None:
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

        with torch.no_grad():  # Disable gradient calculations
            for batch_idx, batch in enumerate(self.val_dataloader):
                # Move batch to device
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

        try:
            for epoch in range(self.start_epoch, self.config.training.epochs):
                self._run_callbacks("on_epoch_start", epoch=epoch)
                epoch_start_time = time.time()
                avg_loss = self._run_epoch(epoch)
                val_loss = None
                if self.config.training.run_validation:  # TODO: Add run_validation to TrainingConfig
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

                    # Save checkpoint based on validation loss if available, otherwise training loss
                    metric_for_checkpoint = val_loss if val_loss is not None else avg_loss
                    self.checkpoint_manager.save_checkpoint(
                        epoch, self.model, self.optimizer, self.scheduler, self.scaler, metric_for_checkpoint
                    )
                    self._run_callbacks("on_save_checkpoint", epoch=epoch)

                logs = {"avg_loss": avg_loss}
                if val_loss is not None:
                    logs["val_loss"] = val_loss
                self._run_callbacks("on_epoch_end", epoch=epoch, logs=logs)

        except Exception as e:
            self._run_callbacks("on_exception", exception=e)
            raise

        finally:
            if self.rank == 0:
                total_time = time.time() - self.training_start_time
                self.logger.info(f"✅ Training completed in {total_time / 3600:.2f} hours on {self.world_size} GPUs.")
                self.logger.info(f"🌟 Best loss achieved: {self.checkpoint_manager.best_loss:.4f}")
            self._run_callbacks("on_train_end")
