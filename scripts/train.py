import os
import time
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


class SimpleMLP(nn.Module):
    def __init__(self, hidden_size=512, ffn_hidden_size=None):
        super().__init__()
        ffn_hidden_size = hidden_size * 4 if ffn_hidden_size is None else ffn_hidden_size
        self.net = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.ReLU(),
            nn.Linear(ffn_hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.net(x)


@contextmanager
def distributed_context(rank, world_size):
    """Context manager for distributed training setup and cleanup"""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12355")

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())

    try:
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def create_dataloader(batch_size=128, num_samples=10000, hidden_size=512):
    x = torch.randn(num_samples, hidden_size)
    y = torch.randn(num_samples, hidden_size)
    dataset = TensorDataset(x, y)

    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    return dataloader, sampler


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.MSELoss()

    for data, target in dataloader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def get_env_config():
    return {
        "master_addr": os.environ.get("MASTER_ADDR", "127.0.0.1"),
        "master_port": os.environ.get("MASTER_PORT", "12355"),
        "num_nodes": int(os.environ.get("NUM_NODES", 1)),
        "gpus_per_node": int(os.environ.get("NUM_GPUS_PER_NODE", torch.cuda.device_count())),
        "node_rank": int(os.environ.get("NODE_RANK", 0)),
        "epochs": int(os.environ.get("EPOCHS", 10)),
    }


def print_config():
    config = get_env_config()

    print("🔧 Distributed Training Configuration:")
    print(f"   📋 Nodes: {config['num_nodes']} | GPUs per node: {config['gpus_per_node']}")
    print(f"   🏷️     Current node: {config['node_rank']} | Epochs: {config['epochs']}")
    print(f"   🌐 Master: {config['master_addr']}:{config['master_port']}")
    print("=" * 60)


def train_worker(local_rank):
    """Training worker for each GPU process"""
    config = get_env_config()
    global_rank = config["node_rank"] * config["gpus_per_node"] + local_rank
    world_size = config["num_nodes"] * config["gpus_per_node"]

    with distributed_context(global_rank, world_size):
        device = torch.device(f"cuda:{local_rank}")

        # Model setup
        model = DDP(SimpleMLP().to(device), device_ids=[local_rank])
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

        # Data setup
        dataloader, sampler = create_dataloader()

        # Training loop
        start_time = time.time()
        for epoch in range(config["epochs"]):
            sampler.set_epoch(epoch)
            avg_loss = train_epoch(model, dataloader, optimizer, device)
            scheduler.step()

            # Synchronize and report progress
            dist.barrier()
            if global_rank == 0:
                elapsed = time.time() - start_time
                lr = scheduler.get_last_lr()[0]
                print(
                    f"📊 Epoch {epoch + 1:2d}/{config['epochs']} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.6f} | Time: {elapsed:.1f}s"
                )

        # Final summary
        if global_rank == 0:
            total_time = time.time() - start_time
            print(f"🎉 Training completed in {total_time:.1f}s on {world_size} GPUs")


def main():
    """Launch distributed training"""
    print_config()
    config = get_env_config()

    mp.spawn(train_worker, nprocs=config["gpus_per_node"], join=True)


if __name__ == "__main__":
    main()
