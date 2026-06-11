import torch
from torch.utils.data import TensorDataset

from llm.data.modules.map_base import SamplerMapDataModule


class SyntheticDataModule(SamplerMapDataModule):
    """DataModule for generating synthetic regression data."""

    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None):
        train_x = torch.randn(self.config.training.num_samples, self.config.model.hidden_size)
        train_y = train_x + 0.1 * torch.randn_like(train_x)
        self.train_dataset = TensorDataset(train_x, train_y)

        val_num_samples = max(1, self.config.training.num_samples // 10)
        val_x = torch.randn(val_num_samples, self.config.model.hidden_size)
        val_y = val_x + 0.1 * torch.randn_like(val_x)
        self.val_dataset = TensorDataset(val_x, val_y)
