import argparse
import logging
import os
import shutil
import sys
import time
import types
import typing
from dataclasses import dataclass, fields
from pathlib import Path

# Rich library for beautiful console output
import rich
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


# ============================================================================
# 1. 扁平化配置 (Simpler Configuration)
# ============================================================================
@dataclass
class TrainingConfig:
    """将所有配置合并到一个类中, 更易于管理和传递"""

    # Model
    hidden_size: int = 512
    ffn_hidden_size: int = 2048
    num_layers: int = 2
    dropout: float = 0.1
    # Training
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    # Data
    num_samples: int = 20000
    # DDP
    backend: str = "nccl"
    # Optimization
    use_compile: bool = True
    use_amp: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    # Checkpointing & Logging
    checkpoint_dir: str = "checkpoints"
    resume_from: str | None = None
    log_dir: str = "logs"


def get_config() -> TrainingConfig:
    """通过 argparse 解析命令行参数来覆盖默认配置"""
    parser = argparse.ArgumentParser(description="A streamlined PyTorch DDP training script with rich logging.")
    config = TrainingConfig()

    for f in fields(config):
        type_for_argparse = f.type
        origin = typing.get_origin(f.type)
        if origin is types.UnionType or origin is typing.Union:
            base_type = next((t for t in typing.get_args(f.type) if t is not types.NoneType), None)
            if base_type:
                type_for_argparse = base_type

        if type_for_argparse is bool:
            if f.default:
                parser.add_argument(
                    f"--no-{f.name.replace('_', '-')}", action="store_false", dest=f.name, help=f"Disable {f.name}"
                )
            else:
                parser.add_argument(
                    f"--{f.name.replace('_', '-')}", action="store_true", dest=f.name, help=f"Enable {f.name}"
                )
        else:
            parser.add_argument(
                f"--{f.name.replace('_', '-')}", type=type_for_argparse, default=None, help=f"Default: {f.default}"
            )

    args = parser.parse_args()

    for f in fields(config):
        arg_val = getattr(args, f.name)
        if arg_val is not None:
            setattr(config, f.name, arg_val)

    return config


# ============================================================================
# 2. 辅助函数 (Utility Functions)
# ============================================================================
def setup_logging(rank: int):
    """设置日志, 主进程使用 RichHandler, 其他进程静默"""
    logger = logging.getLogger(f"train_rank_{rank}")
    logger.propagate = False

    if rank == 0:
        logger.setLevel(logging.INFO)
        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=True)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
    else:
        logger.setLevel(logging.WARNING)
        logger.addHandler(logging.NullHandler())
    return logger


def setup_ddp(rank: int, world_size: int, backend: str):
    """设置 DDP 环境"""
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed_all(42 + rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# 3. 模型 (Model Definition)
# ============================================================================
class SimpleMLP(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, config.ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_hidden_size, config.hidden_size),
        )
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
# 4. 训练器 (The Streamlined Trainer with Rich Integration)
# ============================================================================
class Trainer:
    def __init__(self, config: TrainingConfig, rank: int, world_size: int, logger: logging.Logger):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.logger = logger
        self.device = torch.device(f"cuda:{rank}")

        model = SimpleMLP(self.config).to(self.device)
        if self.config.use_compile and sys.version_info >= (3, 8):
            self.logger.info("Compiling model with torch.compile...")
            model = torch.compile(model)
        self.model = DistributedDataParallel(model, device_ids=[self.rank])

        self.train_loader, self.sampler = self._create_dataloader()

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay, fused=True
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        self.scaler = torch.amp.GradScaler(enabled=self.config.use_amp)
        self.criterion = nn.MSELoss()

        self.start_epoch = 0
        self.best_loss = float("inf")
        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)

    def _create_dataloader(self) -> tuple[DataLoader, DistributedSampler]:
        dataset = TensorDataset(
            torch.randn(self.config.num_samples, self.config.hidden_size),
            torch.randn(self.config.num_samples, self.config.hidden_size),
        )
        sampler = DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
        )
        return loader, sampler

    def _save_checkpoint(self, epoch: int, loss: float):
        if self.rank != 0:
            return
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": epoch,
            "loss": loss,
            "best_loss": self.best_loss,
            "model_state": self.model.module.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
        }

        latest_path = checkpoint_dir / "latest.pt"
        temp_path = checkpoint_dir / "latest.pt.tmp"
        torch.save(state, temp_path)
        shutil.move(temp_path, latest_path)

        if loss < self.best_loss:
            self.best_loss = loss
            self.logger.info(f"🏆 New best model found with loss [bold yellow]{loss:.4f}[/], saving to best.pt")
            best_path = checkpoint_dir / "best.pt"
            shutil.copyfile(latest_path, best_path)

    def _load_checkpoint(self, path: str):
        if not Path(path).exists():
            self.logger.warning(f"Checkpoint {path} not found. Starting from scratch.")
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.model.module.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.logger.info(f"✅ Resumed from checkpoint [cyan]{path}[/] at epoch {self.start_epoch}.")

    def train(self):
        if self.rank == 0:
            self.logger.info(f"🎉 Starting training for {self.config.epochs} epochs...")

        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            TextColumn("[bold blue]Loss: {task.fields[loss]:.4f} | LR: {task.fields[lr]:.2e}"),
        ]

        total_start_time = time.time()
        with Progress(*progress_columns, transient=False) if self.rank == 0 else Path(os.devnull).open("w") as progress:
            epoch_task = (
                progress.add_task("[bold red]Epochs", total=self.config.epochs, loss=0.0, lr=0.0)
                if self.rank == 0
                else None
            )

            for epoch in range(self.start_epoch, self.config.epochs):
                self.sampler.set_epoch(epoch)
                self.model.train()

                epoch_loss = 0.0
                if self.rank == 0:
                    torch.cuda.reset_peak_memory_stats(self.device)

                batch_task = (
                    progress.add_task(f"[cyan]Epoch {epoch + 1}", total=len(self.train_loader), loss=0.0, lr=0.0)
                    if self.rank == 0
                    else None
                )

                for data, target in self.train_loader:
                    data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.config.use_amp):
                        output = self.model(data)
                        loss = self.criterion(output, target)

                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    epoch_loss += loss.item()

                    if self.rank == 0:
                        progress.update(
                            batch_task, advance=1, loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"]
                        )

                self.scheduler.step()

                avg_loss_tensor = torch.tensor(epoch_loss / len(self.train_loader), device=self.device)
                dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
                avg_loss = avg_loss_tensor.item()

                if self.rank == 0:
                    peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024**3
                    self.logger.info(
                        f"Epoch {epoch + 1} Summary | "
                        f"Avg Loss: [bold yellow]{avg_loss:.4f}[/] | "
                        f"Peak Mem: [bold cyan]{peak_mem:.2f} GB[/]"
                    )
                    self._save_checkpoint(epoch, avg_loss)
                    progress.update(epoch_task, advance=1)
                    progress.remove_task(batch_task)

        if self.rank == 0:
            total_time = time.time() - total_start_time
            self.logger.info(f"✅ Training finished in [bold green]{total_time / 3600:.2f} hours[/].")


# ============================================================================
# 5. 主入口 (Main Entrypoint)
# ============================================================================
def worker_fn(rank: int, world_size: int, config: TrainingConfig):
    """每个 DDP 进程执行的函数"""
    logger = setup_logging(rank)
    try:
        setup_ddp(rank, world_size, config.backend)
        trainer = Trainer(config, rank, world_size, logger)
        trainer.train()
    except Exception:
        logger.error(f"FATAL ERROR in rank {rank}", exc_info=True)
    finally:
        cleanup_ddp()


def main():
    config = get_config()

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    world_size = torch.cuda.device_count()
    if world_size == 0:
        rich.print("[bold red]❌ No GPUs found. Exiting.[/]")
        sys.exit(1)

    # 在主进程中打印一次配置信息
    if int(os.environ.get("RANK", 0)) == 0:
        table = Table(title="🔧 Training Configuration", show_header=True, header_style="bold magenta")
        table.add_column("Parameter", style="dim", width=20)
        table.add_column("Value", style="bold")
        for f in fields(config):
            table.add_row(f.name, str(getattr(config, f.name)))
        rich.print(table)

    if world_size > 1:
        rich.print(f"🚀 [bold green]Spawning {world_size} DDP processes...[/]")
        mp.spawn(worker_fn, args=(world_size, config), nprocs=world_size, join=True)
    else:
        rich.print("🚀 [bold green]Starting single-GPU training...[/]")
        worker_fn(0, 1, config)


if __name__ == "__main__":
    main()
