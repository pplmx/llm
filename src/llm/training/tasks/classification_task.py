import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from llm.training.models.simple_classifier import SimpleClassifier
from llm.training.tasks.base_task import TrainingTask


class ClassificationTask(TrainingTask):
    NUM_CLASSES = 10  # Example

    def build_model(self) -> nn.Module:
        return SimpleClassifier(self.config.model, num_classes=self.NUM_CLASSES)

    def build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        # Let's use SGD for this task for variety
        return optim.SGD(model.parameters(), lr=self.config.training.lr, momentum=0.9)

    def build_scheduler(self, optimizer: optim.Optimizer):
        # A different scheduler
        return optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    def build_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def build_dataloader(self, rank: int, world_size: int):
        # Synthetic classification data
        x = torch.randn(self.config.training.num_samples, self.config.model.hidden_size)
        y = torch.randint(0, self.NUM_CLASSES, (self.config.training.num_samples,))
        dataset = TensorDataset(x, y)

        # Dataloader creation logic is often reusable
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            sampler=sampler,
            num_workers=self.config.optimization.num_workers,
            pin_memory=self.config.optimization.pin_memory,
        )
        return dataloader, sampler

    def train_step(self, batch, model: nn.Module, criterion: nn.Module):
        data, target = batch
        logits = model(data)
        loss = criterion(logits, target)

        # We can calculate and return more metrics
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == target).float().mean()

        return loss, {"loss": loss.item(), "accuracy": accuracy.item()}
