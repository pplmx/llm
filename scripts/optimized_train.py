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


# [BEST PRACTICE] Using a dataclass to manage configurations provides clarity and type hints.
@dataclass
class TrainConfig:
    """Configuration settings for the training process."""

    # Model parameters
    hidden_size: int = 512
    ffn_hidden_size: int = 2048

    # Training parameters
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.01

    # Dataset parameters
    num_samples: int = 20000

    # DDP environment parameters
    master_addr: str = "127.0.0.1"
    master_port: str = "12355"
    num_nodes: int = 1
    gpus_per_node: int = torch.cuda.device_count()
    node_rank: int = 0

    # Performance optimization switches
    use_compile: bool = True  # Whether to use torch.compile
    use_amp: bool = True  # Whether to use Automatic Mixed Precision (AMP)


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
    """Initializes the process group for distributed training."""
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())


def cleanup_ddp():
    """Destroys the DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


class Trainer:
    """
    Encapsulates the training logic for better modularity and reusability.
    """

    def __init__(self, config: TrainConfig, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

        model = SimpleMLP(config.hidden_size, config.ffn_hidden_size)

        # [PERFORMANCE] torch.compile (PyTorch 2.0+) can significantly speed up the model.
        # The 'reduce-overhead' mode is a good choice for models with less dynamic shapes.
        if config.use_compile:
            if self.rank == 0:
                print("🚀 Compiling the model with torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead")

        self.model = DDP(model.to(self.device), device_ids=[self.device.index])

        # [PERFORMANCE] The 'fused' option for AdamW can improve GPU performance.
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            fused=torch.cuda.is_available(),
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)

        # GradScaler is used for automatic mixed precision (AMP) to prevent underflow.
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

        self.dataloader, self.sampler = self._create_dataloader()
        self.criterion = nn.MSELoss()

    def _create_dataloader(self):
        # Each rank creates the same synthetic data. The DDP Sampler ensures each
        # process gets a unique, non-overlapping slice of the data.
        x = torch.randn(self.config.num_samples, self.config.hidden_size)
        y = torch.randn(self.config.num_samples, self.config.hidden_size)
        dataset = TensorDataset(x, y)

        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)

        # Using num_workers > 0 and pin_memory=True are best practices for data loading performance.
        dataloader = DataLoader(
            dataset, batch_size=self.config.batch_size, sampler=sampler, num_workers=4, pin_memory=True
        )
        return dataloader, sampler

    def _run_epoch(self, epoch: int):
        # This is crucial to ensure that shuffling is different and properly coordinated across epochs.
        self.sampler.set_epoch(epoch)
        self.model.train()

        total_loss = 0.0
        for data, target in self.dataloader:
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # [PERFORMANCE] set_to_none=True is a micro-optimization that avoids a memory-setting kernel.
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.config.use_amp):
                output = self.model(data)
                loss = self.criterion(output, target)

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

            # A barrier synchronizes all processes, ensuring all ranks have finished
            # the current epoch before the main process prints the log.
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
    try:
        setup_ddp(rank, world_size, config)
        trainer = Trainer(config, rank, world_size)
        trainer.train()
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
    finally:
        cleanup_ddp()


def get_config_from_env_and_args():
    """
    [BEST PRACTICE] Combine argparse and environment variables for flexible configuration.
    """
    parser = argparse.ArgumentParser(description="PyTorch DDP Training Script")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, help="Batch size per GPU.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")

    args = parser.parse_args()

    # Get distributed settings from environment variables, a standard practice
    # in cluster environments like SLURM.
    config_dict = {
        "node_rank": int(os.environ.get("NODE_RANK", 0)),
        "num_nodes": int(os.environ.get("NUM_NODES", 1)),
        "master_addr": os.environ.get("MASTER_ADDR", "127.0.0.1"),
        "master_port": os.environ.get("MASTER_PORT", "12355"),
        "gpus_per_node": int(os.environ.get("GPUS_PER_NODE", torch.cuda.device_count())),
    }

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
    config = get_config_from_env_and_args()

    # [PERFORMANCE] Enables cudnn's auto-tuner to find the best algorithm for the
    # current hardware. Recommended when input tensor sizes are consistent.
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
