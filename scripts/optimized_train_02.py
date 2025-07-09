import argparse
import logging
import os
import shutil
import sys
import time
import types
import typing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

# ============================================================================
# 配置管理
# ============================================================================


@dataclass
class ModelConfig:
    """模型配置"""

    hidden_size: int = 512
    # REFACTOR: 允许ffn_hidden_size为None，在__post_init__中设置默认值
    ffn_hidden_size: int | None = None
    num_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self):
        # OPTIMIZATION: 将依赖逻辑内聚到配置类自身
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size * 4

        if self.hidden_size <= 0:
            raise ValueError("Hidden size must be positive")
        if self.ffn_hidden_size <= 0:
            raise ValueError("FFN hidden size must be positive")
        if self.num_layers <= 0:
            raise ValueError("Number of layers must be positive")


@dataclass
class TrainingConfig:
    """训练配置"""

    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.01
    num_samples: int = 20000
    scheduler_type: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 1
    gradient_clip_val: float = 1.0

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.lr <= 0:
            raise ValueError("Learning rate must be positive")
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")


@dataclass
class DistributedConfig:
    """分布式配置"""

    master_addr: str = "127.0.0.1"
    master_port: str = "12355"
    num_nodes: int = 1
    gpus_per_node: int = field(default_factory=lambda: torch.cuda.device_count())
    node_rank: int = 0
    backend: str = "nccl"

    def __post_init__(self):
        available_gpus = torch.cuda.device_count()
        if self.gpus_per_node > available_gpus:
            raise ValueError(f"Requested {self.gpus_per_node} GPUs but only {available_gpus} available")


@dataclass
class OptimizationConfig:
    """性能优化配置"""

    use_compile: bool = True
    use_amp: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True

    def __post_init__(self):
        if self.num_workers < 0:
            raise ValueError("Number of workers must be non-negative")


@dataclass
class CheckpointConfig:
    """检查点配置"""

    checkpoint_dir: str = "checkpoints"
    resume_from_checkpoint: str | None = None
    save_interval: int = 1
    keep_last_n: int = 5  # 保留最近N个检查点
    save_best: bool = True

    def __post_init__(self):
        if self.save_interval <= 0:
            raise ValueError("Save interval must be positive")
        if self.keep_last_n <= 0:
            raise ValueError("Keep last N must be positive")


@dataclass
class LoggingConfig:
    """日志配置"""

    log_interval: int = 10  # 每隔多少个batch记录一次
    log_level: str = "INFO"
    log_dir: str = "logs"
    save_logs: bool = True


@dataclass
class Config:
    """主配置类，组合所有配置"""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_args_and_env(cls) -> "Config":
        """从命令行参数和环境变量创建配置"""
        parser = argparse.ArgumentParser(description="PyTorch DDP Training Script")

        # 动态添加参数，避免重复
        def add_args_from_dataclass(parser_, dc_name, dc_instance):
            for name, type_hint in dc_instance.__annotations__.items():
                # 简化处理，只暴露部分关键参数
                arg_name = f"--{name.replace('_', '-')}"
                if name in [
                    "epochs",
                    "batch_size",
                    "lr",
                    "hidden_size",
                    "scheduler_type",
                    "num_workers",
                    "resume_from_checkpoint",
                    "checkpoint_dir",
                    "log_interval",
                    "log_level",
                ]:
                    # 从联合类型 (e.g., str | None) 中提取基础类型 (e.g., str)
                    type_for_argparse = type_hint
                    origin = typing.get_origin(type_hint)
                    if origin is types.UnionType or origin is typing.Union:
                        # 从 (str, NoneType) 中找到非 None 的类型
                        base_type = next((t for t in typing.get_args(type_hint) if t is not types.NoneType), None)
                        if base_type:
                            type_for_argparse = base_type
                        else:
                            # 如果参数类型是 Optional[None] 这种无法从命令行设置的，就跳过
                            continue

                    parser_.add_argument(
                        arg_name, type=type_for_argparse, default=None, help=f"Override {dc_name}.{name}"
                    )

        config = cls()
        add_args_from_dataclass(parser, "training", config.training)
        add_args_from_dataclass(parser, "model", config.model)
        add_args_from_dataclass(parser, "optimization", config.optimization)
        add_args_from_dataclass(parser, "checkpoint", config.checkpoint)
        add_args_from_dataclass(parser, "logging", config.logging)

        parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
        parser.add_argument("--no-amp", action="store_true", help="Disable AMP")

        args = parser.parse_args()

        # 从环境变量更新分布式配置
        config.distributed.node_rank = int(os.environ.get("NODE_RANK", config.distributed.node_rank))
        config.distributed.num_nodes = int(os.environ.get("NUM_NODES", config.distributed.num_nodes))
        config.distributed.master_addr = os.environ.get("MASTER_ADDR", config.distributed.master_addr)
        config.distributed.master_port = os.environ.get("MASTER_PORT", config.distributed.master_port)
        config.distributed.gpus_per_node = int(os.environ.get("GPUS_PER_NODE", torch.cuda.device_count()))

        # REFACTOR: 简化从命令行参数更新配置的逻辑
        for group_name, group_config in config.__dict__.items():
            if group_name.startswith("_"):
                continue
            for key, _ in group_config.__annotations__.items():
                arg_val = getattr(args, key, None)
                if arg_val is not None:
                    setattr(group_config, key, arg_val)

        if args.no_compile:
            config.optimization.use_compile = False
        if args.no_amp:
            config.optimization.use_amp = False

        # 手动后处理
        config.model.__post_init__()
        config.training.__post_init__()

        return config


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
# 模型定义 (无变化，已足够好)
# ============================================================================


class SimpleMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        layers = []
        input_size = config.hidden_size
        for i in range(config.num_layers):
            is_last_layer = i == config.num_layers - 1
            output_size = config.hidden_size if is_last_layer else config.ffn_hidden_size
            layers.append(nn.Linear(input_size, output_size))
            if not is_last_layer:
                layers.append(nn.GELU())  # 使用GELU，现代模型中更常见
                layers.append(nn.Dropout(config.dropout))
            input_size = output_size
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> tuple[int, int]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


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
# 数据管理 (无变化，已足够好)
# ============================================================================


class DataManager:
    def __init__(self, config: Config, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size

    def create_dataloader(self) -> tuple[DataLoader, DistributedSampler]:
        dataset = self._create_dataset()
        sampler = DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True
        )
        use_persistent_workers = (
            self.config.optimization.persistent_workers and self.config.optimization.num_workers > 0
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            sampler=sampler,
            num_workers=self.config.optimization.num_workers,
            pin_memory=self.config.optimization.pin_memory,
            prefetch_factor=self.config.optimization.prefetch_factor if use_persistent_workers else 2,
            persistent_workers=use_persistent_workers,
        )
        return dataloader, sampler

    def _create_dataset(self) -> TensorDataset:
        x = torch.randn(self.config.training.num_samples, self.config.model.hidden_size)
        y = x + 0.1 * torch.randn_like(x)
        return TensorDataset(x, y)


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


# ============================================================================
# 训练器
# ============================================================================


class Trainer:
    def __init__(self, config: Config, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

        self.logger = Logger(rank, config.logging)
        self.performance_monitor = PerformanceMonitor(rank, self.device)
        self.checkpoint_manager = CheckpointManager(config.checkpoint, rank, self.logger)
        self.data_manager = DataManager(config, rank, world_size)

        self._setup_model()
        self._setup_training_components()
        self._setup_data()
        self.training_start_time = time.time()

    def _setup_model(self):
        model = SimpleMLP(self.config.model).to(self.device)
        if self.rank == 0:
            total, trainable = model.count_parameters()
            self.logger.info(f"🏗️  Model: {total:,} total params, {trainable:,} trainable")

        if self.config.optimization.use_compile and sys.version_info >= (3, 8):
            self.logger.info("🚀 Compiling model with torch.compile...")
            try:
                # DDP after compilation is the recommended practice
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}. Continuing without it.")

        # DDP wrapper
        # find_unused_parameters=False is a performance gain, but can cause errors
        # if some model parameters are not used in the forward pass.
        self.model = DDP(model, device_ids=[self.device.index], find_unused_parameters=False)

    def _setup_training_components(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
            fused=self.device.type == "cuda",
        )
        self.scheduler = self._create_scheduler()
        self.scaler = torch.amp.GradScaler(enabled=(self.config.optimization.use_amp and self.device.type == "cuda"))
        self.criterion = nn.MSELoss()

        self.start_epoch, self.best_loss = self.checkpoint_manager.load_checkpoint(
            self.model.module, self.optimizer, self.scheduler, self.scaler, self.device
        )
        self.checkpoint_manager.best_loss = self.best_loss

    def _create_scheduler(self) -> LRScheduler:
        scheduler_map = {
            "cosine": optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.epochs - self.config.training.warmup_epochs
            ),
            "step": optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, self.config.training.epochs // 3), gamma=0.1
            ),
            "plateau": optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=5, factor=0.5),
        }
        scheduler = scheduler_map.get(self.config.training.scheduler_type)
        if not scheduler:
            raise ValueError(f"Unknown scheduler type: {self.config.training.scheduler_type}")
        if self.config.training.warmup_epochs > 0:
            return optim.lr_scheduler.SequentialLR(
                self.optimizer,
                [
                    optim.lr_scheduler.LinearLR(
                        self.optimizer,
                        start_factor=1e-6,
                        end_factor=1.0,
                        total_iters=self.config.training.warmup_epochs,
                    ),
                    scheduler,
                ],
                milestones=[self.config.training.warmup_epochs],
            )
        return scheduler

    def _setup_data(self):
        self.dataloader, self.sampler = self.data_manager.create_dataloader()
        if self.rank == 0:
            self.logger.info(
                f"📊 Dataset: {len(self.dataloader.dataset):,} samples, {len(self.dataloader)} batches per epoch"
            )

    def _run_epoch(self, epoch: int) -> float:
        self.sampler.set_epoch(epoch)
        self.model.train()
        self.performance_monitor.reset_epoch_stats()

        epoch_loss = 0.0
        num_batches = len(self.dataloader)

        for batch_idx, (data, target) in enumerate(self.dataloader):
            batch_start_time = time.time()
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                output = self.model(data)
                loss = self.criterion(output, target)

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            # OPTIMIZATION: Get grad_norm directly from clip_grad_norm_
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip_val)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_loss = loss.item()
            epoch_loss += batch_loss

            self.performance_monitor.log_batch_time(time.time() - batch_start_time)
            self.performance_monitor.log_loss(batch_loss)
            self.performance_monitor.log_gradient_norm(grad_norm.item())

            if (batch_idx + 1) % self.config.logging.log_interval == 0 and self.rank == 0:
                self._log_batch_stats(epoch, batch_idx, num_batches)

        loss_tensor = torch.tensor(epoch_loss / num_batches, device=self.device)
        global_avg_loss = DistributedManager.reduce_mean(loss_tensor).item()

        return global_avg_loss

    def _log_batch_stats(self, epoch, batch_idx, num_batches):
        lr = self.optimizer.param_groups[0]["lr"]
        mem_alloc, mem_cached = self.performance_monitor.get_current_gpu_memory()
        loss = self.performance_monitor.losses[-1]
        grad_norm = self.performance_monitor.gradient_norms[-1]
        batch_time = self.performance_monitor.get_avg_batch_time()
        log_msg = (
            f"Epoch {epoch + 1:2d} | Batch {batch_idx + 1:4d}/{num_batches} | "
            f"Loss: {loss:.4f} | Grad Norm: {grad_norm:.2f} | "
            f"LR: {lr:.6f} | B-Time: {batch_time * 1000:.1f}ms | "
            f"Mem: {mem_alloc:.2f}/{mem_cached:.2f} GB"
        )
        self.logger.info(log_msg)

    def train(self):
        if self.rank == 0:
            self.logger.info("🎉 Starting training...")

        for epoch in range(self.start_epoch, self.config.training.epochs):
            epoch_start_time = time.time()
            avg_loss = self._run_epoch(epoch)

            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()

            DistributedManager.barrier()

            if self.rank == 0:
                epoch_time = time.time() - epoch_start_time
                lr = self.scheduler.get_last_lr()[0]
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


# ============================================================================
# 主函数和入口点
# ============================================================================


def train_worker(rank: int, world_size: int, config: Config):
    """训练工作进程"""
    distributed_manager = DistributedManager(config.distributed)
    try:
        distributed_manager.setup(rank, world_size)
        trainer = Trainer(config, rank, world_size)
        trainer.train()
    except Exception:
        # 使用根logger记录异常，确保即使自定义logger失败也能看到错误
        logging.getLogger().exception(f"An error occurred in rank {rank}")
        # 抛出异常以终止进程
        raise
    finally:
        distributed_manager.cleanup()


def main():
    config = Config.from_args_and_env()

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    distributed_manager = DistributedManager(config.distributed)
    world_size = distributed_manager.get_world_size()

    logger = Logger(0, config.logging)
    if torch.cuda.is_available() and world_size > 0:
        logger.info("=" * 50)
        logger.info("🔧 TRAINING CONFIGURATION 🔧")
        logger.info("=" * 50)
        logger.info(
            f"  Model        : Hidden={config.model.hidden_size}, FFN={config.model.ffn_hidden_size}, Layers={config.model.num_layers}"
        )
        logger.info(
            f"  Training     : Epochs={config.training.epochs}, BS/GPU={config.training.batch_size}, LR={config.training.lr}"
        )
        logger.info(
            f"  Distributed  : World Size={world_size}, Nodes={config.distributed.num_nodes}, GPUs/Node={config.distributed.gpus_per_node}"
        )
        logger.info(
            f"  Optimization : torch.compile={'✅' if config.optimization.use_compile else '❌'}, AMP={'✅' if config.optimization.use_amp else '❌'}"
        )
        logger.info("=" * 50)

        if world_size > 1:
            logger.info(f"🚀 Spawning {world_size} DDP processes...")
            mp.spawn(train_worker, args=(world_size, config), nprocs=config.distributed.gpus_per_node, join=True)
        else:
            logger.info("🚀 Starting single-GPU training...")
            train_worker(0, 1, config)
    else:
        logger.error("❌ No GPUs found or world_size is zero. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
