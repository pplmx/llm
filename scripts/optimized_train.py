import argparse
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


# [BEST PRACTICE] 使用 dataclass 管理配置，更清晰且支持类型提示
@dataclass
class TrainConfig:
    """训练配置"""

    # 模型参数
    hidden_size: int = 512
    ffn_hidden_size: int = 2048

    # 训练参数
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.01

    # 数据集参数
    num_samples: int = 20000

    # DDP 环境参数
    master_addr: str = "127.0.0.1"
    master_port: str = "12355"
    num_nodes: int = 1
    gpus_per_node: int = torch.cuda.device_count()
    node_rank: int = 0

    # 性能优化开关
    use_compile: bool = True  # 是否使用 torch.compile
    use_amp: bool = True  # 是否使用自动混合精度


class SimpleMLP(nn.Module):
    def __init__(self, hidden_size=512, ffn_hidden_size=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.ReLU(),
            nn.Linear(ffn_hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.net(x)


def setup_ddp(rank: int, world_size: int, config: TrainConfig):
    """
    初始化 DDP 环境
    """
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())


def cleanup_ddp():
    """
    清理 DDP 环境
    """
    if dist.is_initialized():
        dist.destroy_process_group()


class Trainer:
    """
    将训练逻辑封装成一个类，提高代码的模块化和可复用性
    """

    def __init__(self, config: TrainConfig, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

        # 1. 创建模型
        model = SimpleMLP(config.hidden_size, config.ffn_hidden_size)

        # [PERFORMANCE] 使用 torch.compile 优化模型 (PyTorch 2.0+)
        if config.use_compile:
            if self.rank == 0:
                print("🚀 Compiling the model with torch.compile()...")
            # `reduce-overhead` 模式可以减少编译开销，适合动态性较小的模型
            model = torch.compile(model, mode="reduce-overhead")

        self.model = DDP(model.to(self.device), device_ids=[self.device.index])

        # 2. 创建优化器和学习率调度器
        # [PERFORMANCE] 对支持的优化器使用 fused=True 以提高GPU性能
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            fused=torch.cuda.is_available(),  # fused 仅在 CUDA 上可用
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)

        # 3. [PERFORMANCE] 创建混合精度训练的 GradScaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

        # 4. 创建数据加载器
        self.dataloader, self.sampler = self._create_dataloader()

        self.criterion = nn.MSELoss()

    def _create_dataloader(self):
        # 在每个 rank 上创建相同的数据，DDP Sampler 会负责分发
        x = torch.randn(self.config.num_samples, self.config.hidden_size)
        y = torch.randn(self.config.num_samples, self.config.hidden_size)
        dataset = TensorDataset(x, y)

        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        # 推荐 num_workers > 0 以利用多进程加载数据
        dataloader = DataLoader(
            dataset, batch_size=self.config.batch_size, sampler=sampler, num_workers=4, pin_memory=True
        )
        return dataloader, sampler

    def _run_epoch(self, epoch: int):
        self.sampler.set_epoch(epoch)  # 保证每个 epoch 的 shuffle 都不同
        self.model.train()

        total_loss = 0.0
        for data, target in self.dataloader:
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # [PERFORMANCE] 使用 set_to_none=True 略微提升性能
            self.optimizer.zero_grad(set_to_none=True)

            # [PERFORMANCE] 使用自动混合精度 (AMP)
            with torch.amp.autocast("cuda", enabled=self.config.use_amp):
                output = self.model(data)
                loss = self.criterion(output, target)

            # 使用 GradScaler 缩放损失并反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)

    def train(self):
        if self.rank == 0:
            print("🎉 Starting training...")

        start_time = time.time()
        for epoch in range(self.config.epochs):
            avg_loss = self._run_epoch(epoch)
            self.scheduler.step()

            # 等待所有进程完成当前 epoch
            dist.barrier()

            if self.rank == 0:
                elapsed = time.time() - start_time
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"📊 Epoch {epoch + 1:2d}/{self.config.epochs} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.6f} | Time: {elapsed:.1f}s"
                )

        if self.rank == 0:
            total_time = time.time() - start_time
            print(f"✅ Training completed in {total_time:.1f}s on {self.world_size} GPUs.")


def train_worker(rank: int, world_size: int, config: TrainConfig):
    """
    为每个 DDP 进程执行的 worker 函数
    """
    try:
        setup_ddp(rank, world_size, config)
        trainer = Trainer(config, rank, world_size)
        trainer.train()
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        # 可以在这里添加更详细的错误处理
    finally:
        cleanup_ddp()


def get_config_from_env_and_args():
    """
    [BEST PRACTICE] 结合 argparse 和环境变量来配置，更加灵活
    """
    parser = argparse.ArgumentParser(description="PyTorch DDP Training Script")
    # 允许命令行覆盖默认配置
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, help="Batch size per GPU.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")

    args = parser.parse_args()

    # 从环境变量获取分布式设置，这是在集群环境中（如 SLURM）的标准做法
    config_dict = {
        "node_rank": int(os.environ.get("NODE_RANK", 0)),
        "num_nodes": int(os.environ.get("NUM_NODES", 1)),
        "master_addr": os.environ.get("MASTER_ADDR", "127.0.0.1"),
        "master_port": os.environ.get("MASTER_PORT", "12355"),
        "gpus_per_node": int(os.environ.get("GPUS_PER_NODE", torch.cuda.device_count())),
    }

    # 命令行参数优先于默认值
    if args.epochs is not None:
        config_dict["epochs"] = args.epochs
    if args.batch_size is not None:
        config_dict["batch_size"] = args.batch_size
    if args.lr is not None:
        config_dict["lr"] = args.lr
    if args.no_compile:
        config_dict["use_compile"] = False

    return TrainConfig(**config_dict)


def main():
    """主函数，启动分布式训练"""
    config = get_config_from_env_and_args()

    # [PERFORMANCE] 启用 CuDNN benchmark 模式
    torch.backends.cudnn.benchmark = True

    world_size = config.num_nodes * config.gpus_per_node

    if world_size > 1:
        print("🔧 Distributed Training Configuration:")
        print(f"   🌍 World Size: {world_size} | Nodes: {config.num_nodes} | GPUs per node: {config.gpus_per_node}")
        print(f"   🏷️   Current node rank: {config.node_rank}")
        print(f"   🌐 Master: {config.master_addr}:{config.master_port}")
        print(f"   ⚙️  torch.compile: {'Enabled' if config.use_compile else 'Disabled'}")
        print(f"   ⚡ AMP: {'Enabled' if config.use_amp else 'Disabled'}")
        print("=" * 60)

        mp.spawn(train_worker, args=(world_size, config), nprocs=config.gpus_per_node, join=True)
    elif world_size == 1:
        print("🚀 Starting single-GPU training...")
        train_worker(0, 1, config)
    else:
        print("No GPUs found. Exiting.")


if __name__ == "__main__":
    main()
