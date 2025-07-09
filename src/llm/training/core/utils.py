import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LRScheduler

from llm.training.core.config import CheckpointConfig, DistributedConfig, LoggingConfig

# ============================================================================
# 性能监控
# ============================================================================


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, rank: int, device: torch.device):
        self.rank = rank
        self.device = device
        self.batch_times: list[float] = []
        self.losses: list[float] = []
        self.gradient_norms: list[float] = []

    def log_batch_time(self, time_taken: float):
        self.batch_times.append(time_taken)

    def log_loss(self, loss: float):
        self.losses.append(loss)

    def log_gradient_norm(self, grad_norm: float):
        self.gradient_norms.append(grad_norm)

    def get_avg_batch_time(self) -> float:
        return sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0.0

    def get_current_gpu_memory(self) -> tuple[float, float]:
        if self.device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            return memory_allocated, memory_reserved
        return 0.0, 0.0

    # OPTIMIZATION: 添加峰值内存监控
    def get_peak_gpu_memory(self) -> float:
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / 1024**3
        return 0.0

    def reset_epoch_stats(self):
        self.batch_times.clear()
        self.losses.clear()
        self.gradient_norms.clear()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)


# ============================================================================
# 日志管理
# ============================================================================


class Logger:
    """增强的日志管理器"""

    def __init__(self, rank: int, config: LoggingConfig):
        self.rank = rank
        self.config = config
        self.logger = logging.getLogger(f"rank_{rank}")
        self._setup_logging()

    def _setup_logging(self):
        self.logger.setLevel(getattr(logging, self.config.log_level.upper()))
        formatter = logging.Formatter(
            f"[%(asctime)s] [%(levelname)s] [Rank {self.rank}] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        if self.rank == 0:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            if self.config.save_logs:
                Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
                # OPTIMIZATION: 使用更友好的时间格式
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                file_handler = logging.FileHandler(os.path.join(self.config.log_dir, f"training_{timestamp}.log"))
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
        else:
            self.logger.addHandler(logging.NullHandler())

    def __getattr__(self, name):
        # 代理所有日志方法 (info, warning, error, etc.)
        return getattr(self.logger, name)


# ============================================================================
# 分布式训练管理 (无变化，已足够好)
# ============================================================================


class DistributedManager:
    def __init__(self, config: DistributedConfig):
        self.config = config

    def setup(self, rank: int, world_size: int):
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = self.config.master_port
        dist.init_process_group(backend=self.config.backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank % torch.cuda.device_count())
        torch.manual_seed(42 + rank)
        torch.cuda.manual_seed_all(42 + rank)

    @staticmethod
    def cleanup():
        if dist.is_initialized():
            dist.destroy_process_group()

    def get_world_size(self) -> int:
        return self.config.num_nodes * self.config.gpus_per_node

    @staticmethod
    def barrier():
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= dist.get_world_size()
        return tensor


# ============================================================================
# 检查点管理
# ============================================================================


class CheckpointManager:
    def __init__(self, config: CheckpointConfig, rank: int, logger: Logger):
        self.config = config
        self.rank = rank
        self.logger = logger
        self.best_loss = float("inf")
        self.checkpoints_saved = []
        if self.rank == 0:
            Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        epoch: int,
        model: DDP,
        optimizer: optim.Optimizer,
        scheduler: LRScheduler,
        scaler: torch.amp.GradScaler,
        loss: float,
    ):
        if self.rank != 0:
            return

        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "best_loss": self.best_loss,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
        }

        # OPTIMIZATION: 原子化保存，先保存到临时文件再移动
        def atomic_save(data, path):
            temp_path = str(path) + ".tmp"
            torch.save(data, temp_path)
            shutil.move(temp_path, path)

        # 保存最新检查点
        latest_path = Path(self.config.checkpoint_dir) / "latest.pt"
        atomic_save(checkpoint, latest_path)

        # 保存周期性检查点
        if (epoch + 1) % self.config.save_interval == 0:
            epoch_path = Path(self.config.checkpoint_dir) / f"epoch_{epoch + 1}.pt"
            atomic_save(checkpoint, epoch_path)
            self.checkpoints_saved.append(epoch_path)
            self._cleanup_old_checkpoints()
            self.logger.debug(f"Checkpoint saved to {epoch_path}")

        # 保存最佳模型
        if self.config.save_best and loss < self.best_loss:
            self.best_loss = loss
            best_path = Path(self.config.checkpoint_dir) / "best.pt"
            checkpoint["best_loss"] = self.best_loss
            atomic_save(checkpoint, best_path)
            self.logger.info(f"🏆 New best model saved with loss {loss:.4f}")

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
        if not os.path.exists(ckp_path):
            self.logger.warning(f"Checkpoint file not found: {ckp_path}. Starting from scratch.")
            return 0, float("inf")

        try:
            map_location = device
            checkpoint = torch.load(ckp_path, map_location=map_location)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint.get("best_loss", float("inf"))
            self.best_loss = best_loss
            self.logger.info(f"✅ Resumed training from epoch {start_epoch} using checkpoint {ckp_path}")
            return start_epoch, best_loss
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {ckp_path}: {e}")
            self.logger.warning("Starting from scratch due to checkpoint loading error.")
            return 0, float("inf")
