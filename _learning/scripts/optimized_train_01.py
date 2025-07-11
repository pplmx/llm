"""
优化后的PyTorch分布式训练脚本
结构更清晰，模块化程度更高，易于维护和扩展
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

# ============================================================================
# 配置管理
# ============================================================================


@dataclass
class ModelConfig:
    """模型配置"""

    hidden_size: int = 512
    ffn_hidden_size: int = 2048


@dataclass
class TrainingConfig:
    """训练配置"""

    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.01
    num_samples: int = 20000


@dataclass
class DistributedConfig:
    """分布式配置"""

    master_addr: str = "127.0.0.1"
    master_port: str = "12355"
    num_nodes: int = 1
    gpus_per_node: int = torch.cuda.device_count()
    node_rank: int = 0


@dataclass
class OptimizationConfig:
    """性能优化配置"""

    use_compile: bool = True
    use_amp: bool = True
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class CheckpointConfig:
    """检查点配置"""

    checkpoint_dir: str = "checkpoints"
    resume_from_checkpoint: str | None = None
    save_interval: int = 1  # 每隔多少个epoch保存一次


@dataclass
class Config:
    """主配置类，组合所有配置"""

    model: ModelConfig
    training: TrainingConfig
    distributed: DistributedConfig
    optimization: OptimizationConfig
    checkpoint: CheckpointConfig

    @classmethod
    def from_args_and_env(cls) -> "Config":
        """从命令行参数和环境变量创建配置"""
        parser = argparse.ArgumentParser(description="PyTorch DDP Training Script")

        # 训练参数
        parser.add_argument("--epochs", type=int, help="Number of training epochs")
        parser.add_argument("--batch-size", type=int, help="Batch size per GPU")
        parser.add_argument("--lr", type=float, help="Learning rate")
        parser.add_argument("--hidden-size", type=int, help="Hidden size of the model")

        # 优化参数
        parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
        parser.add_argument("--no-amp", action="store_true", help="Disable AMP")

        # 检查点参数
        parser.add_argument("--resume-from", type=str, help="Path to checkpoint to resume from")
        parser.add_argument("--checkpoint-dir", type=str, help="Directory to save checkpoints")

        args = parser.parse_args()

        # 创建默认配置
        config = cls(
            model=ModelConfig(),
            training=TrainingConfig(),
            distributed=DistributedConfig(),
            optimization=OptimizationConfig(),
            checkpoint=CheckpointConfig(),
        )

        # 从环境变量更新分布式配置
        config.distributed.node_rank = int(os.environ.get("NODE_RANK", 0))
        config.distributed.num_nodes = int(os.environ.get("NUM_NODES", 1))
        config.distributed.master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        config.distributed.master_port = os.environ.get("MASTER_PORT", "12355")
        config.distributed.gpus_per_node = int(os.environ.get("GPUS_PER_NODE", torch.cuda.device_count()))

        # 从命令行参数更新配置
        if args.epochs is not None:
            config.training.epochs = args.epochs
        if args.batch_size is not None:
            config.training.batch_size = args.batch_size
        if args.lr is not None:
            config.training.lr = args.lr
        if args.hidden_size is not None:
            config.model.hidden_size = args.hidden_size
            config.model.ffn_hidden_size = args.hidden_size * 4
        if args.no_compile:
            config.optimization.use_compile = False
        if args.no_amp:
            config.optimization.use_amp = False
        if args.resume_from:
            config.checkpoint.resume_from_checkpoint = args.resume_from
        if args.checkpoint_dir:
            config.checkpoint.checkpoint_dir = args.checkpoint_dir

        return config


# ============================================================================
# 日志管理
# ============================================================================


class Logger:
    """日志管理器"""

    def __init__(self, rank: int):
        self.rank = rank
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        """配置日志"""
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(f"[%(asctime)s] [%(levelname)s] [Rank {self.rank}] %(message)s")

        if self.rank == 0:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        else:
            self.logger.addHandler(logging.NullHandler())

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def exception(self, msg: str):
        self.logger.exception(msg)


# ============================================================================
# 模型定义
# ============================================================================


class SimpleMLP(nn.Module):
    """简单的多层感知机模型"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, config.ffn_hidden_size),
            nn.ReLU(),
            nn.Linear(config.ffn_hidden_size, config.hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# 分布式训练管理
# ============================================================================


class DistributedManager:
    """分布式训练管理器"""

    def __init__(self, config: DistributedConfig):
        self.config = config

    def setup(self, rank: int, world_size: int):
        """初始化分布式训练环境"""
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = self.config.master_port

        # 使用NCCL后端进行GPU通信
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank % torch.cuda.device_count())

    @staticmethod
    def cleanup():
        """清理分布式训练环境"""
        if dist.is_initialized():
            dist.destroy_process_group()

    def get_world_size(self) -> int:
        """获取总的进程数"""
        return self.config.num_nodes * self.config.gpus_per_node


# ============================================================================
# 数据管理
# ============================================================================


class DataManager:
    """数据管理器"""

    def __init__(self, config: Config, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size

    def create_dataloader(self) -> tuple[DataLoader, DistributedSampler]:
        """创建数据加载器"""
        # 创建合成数据集
        x = torch.randn(self.config.training.num_samples, self.config.model.hidden_size)
        y = torch.randn(self.config.training.num_samples, self.config.model.hidden_size)
        dataset = TensorDataset(x, y)

        # 创建分布式采样器
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)

        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            sampler=sampler,
            num_workers=self.config.optimization.num_workers,
            pin_memory=self.config.optimization.pin_memory,
        )

        return dataloader, sampler


# ============================================================================
# 检查点管理
# ============================================================================


class CheckpointManager:
    """检查点管理器"""

    def __init__(self, config: CheckpointConfig, rank: int, logger: Logger):
        self.config = config
        self.rank = rank
        self.logger = logger

    def save_checkpoint(
        self, epoch: int, model: DDP, optimizer: optim.Optimizer, scaler: torch.amp.GradScaler, loss: float
    ):
        """保存检查点"""
        if self.rank != 0:
            return

        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"epoch_{epoch}.pt")

        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.debug(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(
        self, model: nn.Module, optimizer: optim.Optimizer, scaler: torch.amp.GradScaler, rank: int
    ) -> int:
        """加载检查点，返回起始epoch"""
        if not self.config.resume_from_checkpoint:
            return 0

        ckp_path = self.config.resume_from_checkpoint
        if not os.path.exists(ckp_path):
            self.logger.warning(f"Checkpoint file not found: {ckp_path}. Starting from scratch.")
            return 0

        map_location = {"cuda:0": f"cuda:{rank}"}
        checkpoint = torch.load(ckp_path, map_location=map_location)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])

        start_epoch = checkpoint["epoch"] + 1
        self.logger.info(f"Resumed training from epoch {start_epoch} using checkpoint {ckp_path}")

        return start_epoch


# ============================================================================
# 训练器
# ============================================================================


class Trainer:
    """训练器主类"""

    def __init__(self, config: Config, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

        # 初始化各个管理器
        self.logger = Logger(rank)
        self.checkpoint_manager = CheckpointManager(config.checkpoint, rank, self.logger)
        self.data_manager = DataManager(config, rank, world_size)

        # 初始化模型和训练组件
        self._setup_model()
        self._setup_training_components()
        self._setup_data()

    def _setup_model(self):
        """设置模型"""
        # 创建模型
        model = SimpleMLP(self.config.model)

        # 加载检查点（在DDP包装之前）
        self.start_epoch = self.checkpoint_manager.load_checkpoint(model, None, None, self.rank)

        # 移动到设备
        model = model.to(self.device)

        # 编译模型（如果启用）
        if self.config.optimization.use_compile:
            self.logger.info("🚀 Compiling model with torch.compile...")
            model = torch.compile(model, mode="reduce-overhead")

        # 包装为DDP
        self.model = DDP(model, device_ids=[self.device.index])

    def _setup_training_components(self):
        """设置训练组件"""
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
            fused=torch.cuda.is_available(),
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.training.epochs)

        # 混合精度缩放器
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.config.optimization.use_amp)

        # 损失函数
        self.criterion = nn.MSELoss()

        # 如果有检查点，重新加载优化器和缩放器状态
        if self.config.checkpoint.resume_from_checkpoint:
            self.start_epoch = self.checkpoint_manager.load_checkpoint(
                self.model.module, self.optimizer, self.scaler, self.rank
            )

    def _setup_data(self):
        """设置数据"""
        self.dataloader, self.sampler = self.data_manager.create_dataloader()

    def _run_epoch(self, epoch: int) -> float:
        """运行一个epoch"""
        self.sampler.set_epoch(epoch)
        self.model.train()
        total_loss = 0.0

        for data, target in self.dataloader:
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # 清零梯度
            self.optimizer.zero_grad(set_to_none=True)

            # 前向传播（使用混合精度）
            with torch.amp.autocast("cuda", enabled=self.config.optimization.use_amp):
                output = self.model(data)
                loss = self.criterion(output, target)

            # 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)

    def train(self):
        """主训练循环"""
        self.logger.info("🎉 Starting training...")
        start_time = time.time()

        for epoch in range(self.start_epoch, self.config.training.epochs):
            # 训练一个epoch
            avg_loss = self._run_epoch(epoch)

            # 更新学习率
            self.scheduler.step()

            # 同步所有进程
            dist.barrier()

            # 日志记录和检查点保存（仅rank 0）
            if self.rank == 0:
                elapsed = time.time() - start_time
                lr = self.scheduler.get_last_lr()[0]

                self.logger.info(
                    f"Epoch {epoch + 1:2d}/{self.config.training.epochs} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.6f} | Time: {elapsed:.1f}s"
                )

                # 保存检查点
                if (epoch + 1) % self.config.checkpoint.save_interval == 0:
                    self.checkpoint_manager.save_checkpoint(epoch, self.model, self.optimizer, self.scaler, avg_loss)

        if self.rank == 0:
            total_time = time.time() - start_time
            self.logger.info(f"✅ Training completed in {total_time:.1f}s on {self.world_size} GPUs.")


# ============================================================================
# 主函数和入口点
# ============================================================================


def train_worker(rank: int, world_size: int, config: Config):
    """训练工作进程"""
    distributed_manager = DistributedManager(config.distributed)

    try:
        # 设置分布式环境
        distributed_manager.setup(rank, world_size)

        # 创建并运行训练器
        trainer = Trainer(config, rank, world_size)
        trainer.train()

    except Exception as e:
        logger = Logger(rank)
        logger.exception(f"An error occurred in rank {rank}: {e}")
        raise
    finally:
        # 清理分布式环境
        distributed_manager.cleanup()


def main():
    """主函数"""
    # 获取配置
    config = Config.from_args_and_env()

    # 性能优化设置
    torch.backends.cudnn.benchmark = True

    # 计算总的进程数
    distributed_manager = DistributedManager(config.distributed)
    world_size = distributed_manager.get_world_size()

    # 设置临时日志（主进程）
    logger = Logger(0)

    if world_size > 1:
        # 多GPU分布式训练
        logger.info("🔧 Distributed Training Configuration:")
        logger.info(
            f"   🌍 World Size: {world_size} | Nodes: {config.distributed.num_nodes} | GPUs per node: {config.distributed.gpus_per_node}"
        )
        logger.info(f"   🏷️  Current node rank: {config.distributed.node_rank}")
        logger.info(f"   🌐 Master: {config.distributed.master_addr}:{config.distributed.master_port}")
        logger.info(f"   ⚙️  torch.compile: {'Enabled' if config.optimization.use_compile else 'Disabled'}")
        logger.info(f"   ⚡ AMP: {'Enabled' if config.optimization.use_amp else 'Disabled'}")

        # 启动多进程训练
        mp.spawn(
            train_worker,
            args=(world_size, config),
            nprocs=config.distributed.gpus_per_node,
            join=True,
        )
    elif world_size == 1:
        # 单GPU训练
        logger.info("🚀 Starting single-GPU training...")
        train_worker(0, 1, config)
    else:
        logger.error("❌ No GPUs found. Exiting.")
        return


if __name__ == "__main__":
    main()
