import sys
import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from llm.training.core.config import Config
from llm.training.core.utils import CheckpointManager, DistributedManager, Logger, PerformanceMonitor
from llm.training.tasks.base_task import TrainingTask


class TrainingEngine:
    def __init__(self, config: Config, task: TrainingTask, rank: int, world_size: int):
        self.config = config
        self.task = task
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

        self.logger = Logger(rank, config.logging)
        self.performance_monitor = PerformanceMonitor(rank, self.device)
        self.checkpoint_manager = CheckpointManager(config.checkpoint, rank, self.logger)

        self._setup_components()
        self.training_start_time = time.time()

    def _setup_components(self):
        """Builds all necessary components for training from the task."""
        self.logger.info("Setting up training components...")

        # Build model and move to device
        model = self.task.build_model().to(self.device)
        if self.rank == 0:
            total, trainable = model.count_parameters()
            self.logger.info(f"🏗️  Model: {total:,} total params, {trainable:,} trainable")

        # Compile model if enabled
        if self.config.optimization.use_compile and sys.version_info >= (3, 8):
            self.logger.info("🚀 Compiling model with torch.compile...")
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}. Continuing without it.")

        # DDP wrapper
        self.model = DDP(model, device_ids=[self.device.index], find_unused_parameters=False)

        # Build other components from task
        self.optimizer = self.task.build_optimizer(self.model)
        self.scheduler = self.task.build_scheduler(self.optimizer)
        self.criterion = self.task.build_criterion().to(self.device)
        self.dataloader, self.sampler = self.task.build_dataloader(self.rank, self.world_size)

        # Setup scaler and load checkpoint
        self.scaler = torch.amp.GradScaler(enabled=(self.config.optimization.use_amp and self.device.type == "cuda"))
        self.start_epoch, self.best_loss = self.checkpoint_manager.load_checkpoint(
            self.model.module, self.optimizer, self.scheduler, self.scaler, self.device
        )
        self.checkpoint_manager.best_loss = self.best_loss

    def _run_epoch(self, epoch: int) -> float:
        self.sampler.set_epoch(epoch)
        self.model.train()
        self.performance_monitor.reset_epoch_stats()

        epoch_loss = 0.0
        num_batches = len(self.dataloader)

        for batch_idx, batch in enumerate(self.dataloader):
            batch_start_time = time.time()
            # Move batch to device
            batch = tuple(t.to(self.device, non_blocking=True) for t in batch)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                loss, metrics = self.task.train_step(batch, self.model, self.criterion)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip_val)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_loss = metrics.get("loss", loss.item())
            epoch_loss += batch_loss

            # Performance monitoring
            self.performance_monitor.log_batch_time(time.time() - batch_start_time)
            self.performance_monitor.log_loss(batch_loss)
            self.performance_monitor.log_gradient_norm(grad_norm.item())

            if (batch_idx + 1) % self.config.logging.log_interval == 0 and self.rank == 0:
                self._log_batch_stats(epoch, batch_idx, num_batches, metrics)

        loss_tensor = torch.tensor(epoch_loss / num_batches, device=self.device)
        global_avg_loss = DistributedManager.reduce_mean(loss_tensor).item()

        return global_avg_loss

    def _log_batch_stats(self, epoch, batch_idx, num_batches, metrics):
        lr = self.optimizer.param_groups[0]["lr"]
        mem_alloc, mem_cached = self.performance_monitor.get_current_gpu_memory()
        loss = metrics.get("loss", 0.0)
        grad_norm = self.performance_monitor.gradient_norms[-1]
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
        if self.rank == 0:
            self.logger.info("🎉 Starting training...")

        for epoch in range(self.start_epoch, self.config.training.epochs):
            epoch_start_time = time.time()
            avg_loss = self._run_epoch(epoch)

            if self.scheduler:
                # ReduceLROnPlateau needs the metric, others don't
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_loss)
                else:
                    self.scheduler.step()

            DistributedManager.barrier()

            if self.rank == 0:
                epoch_time = time.time() - epoch_start_time
                lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]["lr"]
                peak_mem = self.performance_monitor.get_peak_gpu_memory()
                self.logger.info("-" * 80)
                self.logger.info(
                    f"Epoch {epoch + 1:2d}/{self.config.training.epochs} SUMMARY | "
                    f"Avg Loss: {avg_loss:.4f} | LR: {lr:.6f} | "
                    f"Time: {epoch_time:.2f}s | Peak Mem: {peak_mem:.2f} GB"
                )
                self.logger.info("-" * 80)

                self.checkpoint_manager.save_checkpoint(
                    epoch, self.model, self.optimizer, self.scheduler, self.scaler, avg_loss
                )

        if self.rank == 0:
            total_time = time.time() - self.training_start_time
            self.logger.info(f"✅ Training completed in {total_time / 3600:.2f} hours on {self.world_size} GPUs.")
            self.logger.info(f"🌟 Best loss achieved: {self.checkpoint_manager.best_loss:.4f}")
