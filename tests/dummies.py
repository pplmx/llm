import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from llm.data.data_module import BaseDataModule


class DummyLMDataModule(BaseDataModule):
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self, rank, world_size):
        # Create samples
        num_samples = self.config.training.num_samples
        S = self.config.model.max_seq_len
        V = self.config.model.vocab_size

        # input_ids: (num_samples, S)
        input_ids = torch.randint(0, V, (num_samples, S), dtype=torch.long)
        labels = torch.randint(0, V, (num_samples, S), dtype=torch.long)

        dataset = TensorDataset(input_ids, labels)

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

        return DataLoader(dataset, batch_size=self.config.training.batch_size, sampler=sampler, drop_last=True), sampler

    def val_dataloader(self, rank, world_size):
        return None, None
