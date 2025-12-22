import argparse
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

# ============================================================================
# 1. 配置管理 (Configuration)
# ============================================================================


@dataclass
class Config:
    """主配置类, 整合所有参数"""

    # 模型配置
    hidden_size: int = 512
    ffn_hidden_size: int | None = None  # 若为None, 则自动设置为 hidden_size * 4
    num_layers: int = 2
    dropout: float = 0.1

    # 训练配置
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    num_samples: int = 20000

    # 优化配置
    use_amp: bool = True
    use_compile: bool = True
    num_workers: int = 4
    pin_memory: bool = True

    # 检查点配置
    checkpoint_dir: str = "checkpoints"
    resume_from_checkpoint: str | None = None
    save_interval: int = 1
    keep_last_n: int = 3
    save_best: bool = True

    # 日志配置
    log_interval: int = 50  # 每N个batch记录一次

    # 分布式配置 (主要通过环境变量设置)
    backend: str = "nccl"
    master_addr: str = field(init=False)
    master_port: str = field(init=False)
    world_size: int = field(init=False)
    rank: int = field(init=False)

    def __post_init__(self):
        """在初始化后进行依赖计算和验证"""
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size * 4

    @classmethod
    def from_args(cls) -> "Config":
        """从命令行参数创建配置, 并从环境变量加载分布式设置"""
        parser = argparse.ArgumentParser(description="An elegant PyTorch DDP Training Script")
        # 从Config类的字段动态添加命令行参数
        for name, default in cls.__annotations__.items():
            if name in ["master_addr", "master_port", "world_size", "rank"]:
                continue  # 这些从环境变量获取
            arg_name = f"--{name.replace('_', '-')}"
            try:
                # 处理 Optional[type] 的情况
                arg_type = get_origin(default) or default
                if get_origin(arg_type) is Union:
                    arg_type = get_args(arg_type)[0]

                # 为布尔值设置 action
                if arg_type is bool:
                    parser.add_argument(arg_name, action="store_true", default=None)
                else:
                    parser.add_argument(arg_name, type=arg_type)
            except (TypeError, AttributeError):
                continue  # 跳过无法解析的类型

        args = parser.parse_args()

        # 创建默认配置实例
        config = cls()

        # 用命令行参数覆盖默认值
        for name, value in vars(args).items():
            if value is not None:
                setattr(config, name, value)

        # 从环境变量加载分布式配置
        config.master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        config.master_port = os.environ.get("MASTER_PORT", "12355")
        config.world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
        config.rank = int(os.environ.get("RANK", -1))  # -1 表示非分布式环境

        return config


# ============================================================================
# 2. 日志与分布式工具 (Utilities)
# ============================================================================


def setup_logger(rank: int) -> logging.Logger:
    """配置日志记录器"""
    logger = logging.getLogger(f"rank_{rank}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(f"[%(asctime)s][Rank {rank}] %(message)s", datefmt="%H:%M:%S")

    if rank == 0:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def setup_distributed(rank: int, world_size: int, config: Config):
    """初始化分布式环境"""
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = config.master_port
    dist.init_process_group(backend=config.backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.manual_seed(42 + rank)  # 保证每个进程有不同的随机种子


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# 3. 模型定义 (Model)
# ============================================================================


class SimpleMLP(nn.Module):
    """更具代表性的多层感知机模型"""

    def __init__(self, config: Config):
        super().__init__()
        layers = []
        input_size = config.hidden_size
        for i in range(config.num_layers):
            is_last_layer = i == config.num_layers - 1
            output_size = config.hidden_size if is_last_layer else config.ffn_hidden_size
            layers.append(nn.Linear(input_size, output_size))
            if not is_last_layer:
                layers.append(nn.GELU())  # 使用更现代的激活函数
                layers.append(nn.Dropout(config.dropout))
            input_size = output_size
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# 4. 检查点管理 (Checkpointing)
# ============================================================================


class CheckpointManager:
    def __init__(self, config: Config, rank: int, logger: logging.Logger):
        self.config = config
        self.rank = rank
        self.logger = logger
        self.best_metric = float("inf")
        self.saved_checkpoints = []
        if self.rank == 0:
            Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def save(
        self,
        epoch: int,
        metric: float,
        model: DDP,
        optimizer: optim.Optimizer,
        scheduler: LRScheduler,
        scaler: torch.amp.GradScaler,
    ):
        if self.rank != 0:
            return

        state = {
            "epoch": epoch,
            "metric": metric,
            "best_metric": self.best_metric,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
        }

        def atomic_save(data: dict[str, Any], path: Path):
            temp_path = path.with_suffix(".tmp")
            torch.save(data, temp_path)
            shutil.move(temp_path, path)

        # 1. 保存周期性检查点
        if (epoch + 1) % self.config.save_interval == 0:
            epoch_path = Path(self.config.checkpoint_dir) / f"epoch_{epoch + 1}.pt"
            atomic_save(state, epoch_path)
            self.saved_checkpoints.append(epoch_path)
            self._cleanup_old_checkpoints()
            self.logger.info(f"💾 Checkpoint saved to {epoch_path}")

        # 2. 保存最佳模型
        if self.config.save_best and metric < self.best_metric:
            self.best_metric = metric
            state["best_metric"] = self.best_metric
            best_path = Path(self.config.checkpoint_dir) / "best.pt"
            atomic_save(state, best_path)
            self.logger.info(f"🏆 New best model saved with metric {metric:.4f}")

    def _cleanup_old_checkpoints(self):
        while len(self.saved_checkpoints) > self.config.keep_last_n:
            oldest = self.saved_checkpoints.pop(0)
            if oldest.exists():
                oldest.unlink()

    def load(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: LRScheduler,
        scaler: torch.amp.GradScaler,
        device: torch.device,
    ) -> int:
        if not self.config.resume_from_checkpoint:
            return 0

        ckp_path = Path(self.config.resume_from_checkpoint)
        if not ckp_path.exists():
            self.logger.warning(f"Checkpoint file not found: {ckp_path}. Starting from scratch.")
            return 0

        try:
            checkpoint = torch.load(ckp_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
            self.best_metric = checkpoint.get("best_metric", float("inf"))
            start_epoch = checkpoint["epoch"] + 1
            self.logger.info(f"✅ Resumed from epoch {start_epoch} using {ckp_path}")
            return start_epoch
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {ckp_path}: {e}. Starting from scratch.")
            return 0


# ============================================================================
# 5. 训练器 (Trainer)
# ============================================================================


class Trainer:
    def __init__(self, config: Config, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        self.logger = setup_logger(rank)

        # 1. 初始化模型
        model = SimpleMLP(self.config).to(self.device)

        # 2. 初始化优化器、调度器、AMP Scaler
        self.optimizer = optim.AdamW(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay, fused=True
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        self.scaler = torch.amp.GradScaler(enabled=self.config.use_amp)
        self.criterion = nn.MSELoss()

        # 3. 加载检查点(在DDP包装前)
        self.checkpoint_manager = CheckpointManager(config, rank, self.logger)
        self.start_epoch = self.checkpoint_manager.load(model, self.optimizer, self.scheduler, self.scaler, self.device)

        # 4. 编译模型(在DDP包装前)
        if self.config.use_compile:
            self.logger.info("🚀 Compiling model with torch.compile...")
            model = torch.compile(model, mode="reduce-overhead")

        # 5. DDP包装
        self.model = DDP(model, device_ids=[self.rank])

        # 6. 准备数据
        self.dataloader, self.sampler = self._create_dataloader()

    def _create_dataloader(self) -> tuple[DataLoader, DistributedSampler]:
        x = torch.randn(self.config.num_samples, self.config.hidden_size)
        y = torch.randn_like(x)
        dataset = TensorDataset(x, y)
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
        ), sampler

    def _run_epoch(self, epoch: int) -> float:
        self.sampler.set_epoch(epoch)
        self.model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(self.dataloader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.config.use_amp):
                output = self.model(data)
                loss = self.criterion(output, target)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            # Unscale and clip gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            if self.rank == 0 and (batch_idx + 1) % self.config.log_interval == 0:
                mem = torch.cuda.memory_allocated(self.device) / 1024**3
                self.logger.info(
                    f"Epoch {epoch + 1:2d} | Batch {batch_idx + 1:4d}/{len(self.dataloader)} | "
                    f"Loss: {loss.item():.4f} | Mem (GB): {mem:.2f}"
                )

        return total_loss / len(self.dataloader)

    def train(self):
        if self.rank == 0:
            self.logger.info("🎉 Starting training...")

        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_start_time = time.time()

            avg_loss = self._run_epoch(epoch)

            # 在所有进程上同步平均损失, 以便进行一致的决策
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            global_avg_loss = loss_tensor.item()

            self.scheduler.step()
            dist.barrier()  # 等待所有进程完成epoch

            if self.rank == 0:
                epoch_time = time.time() - epoch_start_time
                lr = self.scheduler.get_last_lr()[0]
                self.logger.info(
                    f"Epoch {epoch + 1:2d} SUMMARY | Avg Loss: {global_avg_loss:.4f} | "
                    f"LR: {lr:.6f} | Time: {epoch_time:.2f}s"
                )
                self.checkpoint_manager.save(
                    epoch, global_avg_loss, self.model, self.optimizer, self.scheduler, self.scaler
                )

        if self.rank == 0:
            self.logger.info(f"✅ Training completed. Best metric: {self.checkpoint_manager.best_metric:.4f}")


# ============================================================================
# 6. 主程序入口 (Main)
# ============================================================================


def worker_fn(rank: int, world_size: int, config: Config):
    """每个DDP进程的执行函数"""
    try:
        setup_distributed(rank, world_size, config)
        trainer = Trainer(config, rank, world_size)
        trainer.train()
    except Exception:
        logging.getLogger(f"rank_{rank}").exception("An error occurred during training.")
        raise
    finally:
        cleanup_distributed()


def main():
    """主函数, 负责启动训练"""
    config = Config.from_args()

    # 设置PyTorch性能选项
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    world_size = config.world_size

    # 临时创建rank 0的日志器用于主进程信息打印
    logger = setup_logger(0)

    if world_size > 1:
        logger.info(f"🚀 Spawning {world_size} DDP processes...")
        # 设置环境变量, 以便worker_fn可以访问
        os.environ["WORLD_SIZE"] = str(world_size)
        mp.spawn(worker_fn, args=(world_size, config), nprocs=world_size, join=True)
    elif world_size == 1 and torch.cuda.is_available():
        logger.info("🚀 Starting single-GPU training...")
        worker_fn(0, 1, config)
    else:
        logger.error("❌ No GPUs found or world_size is zero. Exiting.")
        exit(1)


if __name__ == "__main__":
    main()
