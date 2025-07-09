import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

from llm.core.mlp import MLP
from llm.training.tasks.base_task import TrainingTask


class RegressionTask(TrainingTask):
    """A concrete task for the SimpleMLP regression problem."""

    def build_model(self) -> nn.Module:
        return MLP(self.config.model)

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

    def train_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        data, target = batch
        output = model(data)
        loss = criterion(output, target)
        return loss, {"loss": loss.item()}
