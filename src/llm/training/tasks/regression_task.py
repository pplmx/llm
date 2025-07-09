import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from llm.training.models.simple_mlp import SimpleMLP
from llm.training.tasks.base_task import TrainingTask


class RegressionTask(TrainingTask):
    """A concrete task for the SimpleMLP regression problem."""

    def build_model(self) -> nn.Module:
        return SimpleMLP(self.config.model)

    def build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        return optim.AdamW(
            model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
            fused=torch.cuda.is_available(),
        )

    def build_scheduler(self, optimizer: optim.Optimizer) -> LRScheduler | None:
        scheduler_map = {
            "cosine": optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.training.epochs - self.config.training.warmup_epochs
            ),
            "step": optim.lr_scheduler.StepLR(optimizer, step_size=max(1, self.config.training.epochs // 3), gamma=0.1),
        }
        scheduler = scheduler_map.get(self.config.training.scheduler_type)
        if not scheduler:
            return None
        if self.config.training.warmup_epochs > 0:
            return optim.lr_scheduler.SequentialLR(
                optimizer,
                [
                    optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=self.config.training.warmup_epochs
                    ),
                    scheduler,
                ],
                milestones=[self.config.training.warmup_epochs],
            )
        return scheduler

    def build_criterion(self) -> nn.Module:
        return nn.MSELoss()

    def build_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler]:
        # Create synthetic dataset
        x = torch.randn(self.config.training.num_samples, self.config.model.hidden_size)
        y = x + 0.1 * torch.randn_like(x)
        dataset = TensorDataset(x, y)

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
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

    def train_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        data, target = batch
        output = model(data)
        loss = criterion(output, target)
        return loss, {"loss": loss.item()}
